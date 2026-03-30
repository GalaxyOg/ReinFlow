"""
Parent pre-training agent class.
"""
import os
import random
import numpy as np
from omegaconf import OmegaConf
from util.scheduler import CosineAnnealingWarmupRestarts
import torch
import hydra
import logging
import wandb
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
log = logging.getLogger(__name__)
from util.timer import Timer
from tqdm import tqdm
from agent.pretrain.utils  import batch_to_device
from util.dirs import REINFLOW_DATA_DIR
# from env.gym_utils import make_async
from env.gymnasium_utils import make_async
import imageio
import datetime
# to test in mujoco simulator
class EnvConfig:
    def __init__(self, n_envs, 
                 name, 
                 max_episode_steps, 
                 reset_at_iteration, 
                 save_video, 
                 use_image_obs, 
                 best_reward_threshold_for_success,
                 wrappers, 
                 n_steps, 
                 render, 
                 render_num,
                 robomimic_env_cfg_path:str=None,
                 shape_meta=None):
        self.n_envs = n_envs
        self.name = name
        self.max_episode_steps = max_episode_steps
        self.reset_at_iteration = reset_at_iteration
        self.save_video = save_video
        self.best_reward_threshold_for_success = best_reward_threshold_for_success
        self.wrappers = wrappers
        self.n_steps = n_steps
        self.render_num = render_num 
        self.use_image_obs = use_image_obs  # Change to True if using image observations in the environment
        self.render = render
        self.specific={}  # not implemented for furniture right now.
        self.robomimic_env_cfg_path=robomimic_env_cfg_path
        self.shape_meta=shape_meta

class PreTrainAgent:
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device
        
        # verbose
        self.verbose_train=False
        self.verbose_loss= False
        self.verbose_test= False
        self.test_model_type='ema' # 'original'
        self.test_log_all = False
        self.only_test=False
        
        # Seed
        self.seed = cfg.get("seed", 42)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Wandb
        self.use_wandb = cfg.wandb is not None
        if self.use_wandb:
            # Check if offline mode is enabled (e.g., via config or environment variable)
            offline_mode = cfg.wandb.get("offline_mode", False)  # Add this to your config if desired
            if offline_mode:
                wandb_dir = cfg.wandb.get("dir", "./wandb_offline")  # Local directory for offline logs
            else:
                wandb_dir = cfg.wandb.get("dir", "./wandb")
            os.makedirs(wandb_dir, exist_ok=True)  # Ensure the directory exists
            wandb.init(
                entity=cfg.wandb.entity,
                project=cfg.wandb.project,
                name=cfg.wandb.run,
                config=OmegaConf.to_container(cfg, resolve=True),
                mode="offline" if offline_mode else "online",  # Switch to offline mode
                dir=wandb_dir,  # Specify local directory for offline logs
            )
            # Get the exact subfolder for this run
            run_id = wandb.run.id  # Unique ID for the run
            run_dir = wandb.run.dir  # Full path to the run's directory
            if offline_mode:
                log.info(f"Wandb running in offline mode. Logs will be saved to {os.path.dirname(run_dir)}")
                log.info(f"Run ID: {run_id}")
            else:
                log.info(f"Wandb running online. Run ID: {run_id}")
        # Logging, checkpoints
        self.logdir = cfg.logdir
        self.checkpoint_dir = os.path.join(self.logdir, "checkpoint")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.log_freq = cfg.train.get("log_freq", 1)
        self.save_model_freq = cfg.train.save_model_freq
        self.log_all = cfg.train.get("log_all", True)
        # TensorBoard writer (use instead of wandb when desired)
        # TensorBoard writer (prefer to write into the wandb run directory if wandb is enabled)
        try:
            writer_dir = os.path.join(self.logdir, "tbdir")
            os.makedirs(writer_dir, exist_ok=True)
            self.writer = SummaryWriter(writer_dir)
            log.info(f"TensorBoard writer initialized at {writer_dir}")
        except Exception:
            self.writer = None
        
        # Build model
        self.model = hydra.utils.instantiate(cfg.model)
        self.ema = EMA(cfg.ema)
        self.ema_model = deepcopy(self.model)
        self.print_architecture()
        
        # Training params
        self.n_epochs = cfg.train.n_epochs
        self.batch_size = cfg.train.batch_size
        self.epoch_start_ema = cfg.train.get("epoch_start_ema", 20)
        self.update_ema_freq = cfg.train.get("update_ema_freq", 10)
        self.val_freq = cfg.train.get("val_freq", 100)
        
        # Build dataset
        self.dataset_train = hydra.utils.instantiate(cfg.train_dataset)
        print(f"dataset_train={len(self.dataset_train)}")
        self.dataloader_train = torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=4 if self.dataset_train.device == "cpu" else 0,
            shuffle=True,
            pin_memory=True if self.dataset_train.device == "cpu" else False,
            drop_last=True # revised by ReinFlow Authors when debugging flow-matching shortcut. 
        )
        self.dataloader_val = None
        if "train_split" in cfg.train and cfg.train.train_split < 1:
            val_indices = self.dataset_train.set_train_val_split(cfg.train.train_split)
            self.dataset_val = deepcopy(self.dataset_train)
            self.dataset_val.set_indices(val_indices)
            self.dataloader_val = torch.utils.data.DataLoader(
                self.dataset_val,
                batch_size=self.batch_size,
                num_workers=4 if self.dataset_val.device == "cpu" else 0,
                shuffle=True,
                pin_memory=True if self.dataset_val.device == "cpu" else False,
                drop_last=True # revised by ReinFlow Authors when debugging flow-matching shortcut. 
            )
        
        # optimizer and lr scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
        )
        self.schedule_lr_each_grad_step = cfg.train.get("schedule_lr_each_grad_step", False)
        self.lr_scheduler = CosineAnnealingWarmupRestarts(
            self.optimizer,
            first_cycle_steps=cfg.train.lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=cfg.train.learning_rate,
            min_lr=cfg.train.lr_scheduler.min_lr,
            warmup_steps=cfg.train.lr_scheduler.warmup_steps,
            gamma=1.0,
        )
        self.reset_parameters()

        # resume
        self.epoch = 1
        self.first_epoch = 0
        # when resume from checkpoint, just reduce the train.n_epochs in your configuration file by the amount of epochs you have trained. 
        # our code will automatically restore the checkpoint for the model and the optimizer and the learning rate scheduler and train for another `train.n_epochs` steps. 
        self.resume_path = cfg.get('base_policy_path', None)
        self.resume = self.resume_path is not None
        if self.resume:
            log.info(f"Resuming from {self.resume_path}")
            self.load(epoch=None, custom_path=self.resume_path)
            log.info(f"Resume complete.")

        # Testing in mujoco 
        self.test_in_mujoco = cfg.get('test_in_mujoco', False) # in openai gym envs, we test the training performance in mujoco to save the model with the highest reward
        self.test_freq = cfg.train.get('test_freq', self.n_epochs -1)
        log.info(f"test_in_mujoco=={self.test_in_mujoco}")
        self.render_dir = os.path.join(self.logdir, "render")
        self.result_path = os.path.join(self.logdir, "result.npz")
        os.makedirs(self.render_dir, exist_ok=True)
        self.test_each_step = False
        
        if self.test_in_mujoco:
            self.test_denoising_steps=20 #to be overloaded
            self.test_model_type='ema'   # original
            
            self.best_episode_reward=0.0
            self.success_rate = 0.0
            self.avg_episode_reward = 0.0
            self.avg_episode_reward_std=0.0
            self.avg_episode_length=0.0
            self.avg_episode_length_std=0.0
            self.avg_best_reward=0.0
            self.avg_best_reward_std=0.0
            ############################################ test in mujoco simulator #########################################
            # Create an instance of EnvConfig with the values from the annotated environment
            self.env_name = cfg.env
            env_type = cfg.get("env_type", None)
            act_steps = cfg.horizon_steps # for now
            normalization_path=cfg.get('normalization_path', None) # )
            if not normalization_path:
                raise ValueError(f"Hey you must specify your normalization path if you wish to evaluate periodically during pretraining, but I can't find it in your configuration! Is {os.path.join(REINFLOW_DATA_DIR,'gym',cfg.env,'normalization.npz')} the correct path I guess? Also, make sure to secure that your normalization file truly correspond to your pretraining data, other wise there will be a significant mismatch in preformance. ")
            if cfg.env_suite=='gym':
                # Keep backward compatibility (default=3.0) while allowing
                # task-specific success thresholds (e.g., FFSM uses negative rewards).
                gym_success_threshold = 3.0
                if "eval_env" in cfg and cfg.eval_env.get("best_reward_threshold_for_success", None) is not None:
                    gym_success_threshold = float(cfg.eval_env.best_reward_threshold_for_success)
                elif cfg.get("best_reward_threshold_for_success", None) is not None:
                    gym_success_threshold = float(cfg.best_reward_threshold_for_success)
                env_max_episode_steps=1_000
                rollout_n_steps=5_00
                n_eval_envs=4
                best_reward_threshold_for_success=gym_success_threshold
                robomimic_env_cfg_path=None
                shape_meta=None
                use_image_obs=False
                wrappers={
                        "mujoco_locomotion_lowdim": {
                            "normalization_path": normalization_path
                        },
                        "multi_step": {
                            "n_obs_steps": cfg.cond_steps,
                            "n_action_steps": act_steps,
                            "max_episode_steps": env_max_episode_steps,
                            "reset_within_step": True
                        }
                    }
            elif cfg.env_suite=='robomimic':
                env_max_episode_steps=cfg.eval_env.max_episode_steps
                rollout_n_steps=cfg.eval_env.n_steps
                n_eval_envs=cfg.eval_env.n_envs
                best_reward_threshold_for_success=cfg.eval_env.best_reward_threshold_for_success
                robomimic_env_cfg_path=cfg.robomimic_env_cfg_path
                shape_meta=cfg.get('shape_meta', None)
                use_image_obs=cfg.eval_env.get("use_image_obs", False),
                wrappers=cfg.eval_env.wrappers
            else:
                raise NotImplementedError(f"Sorry about that, we have not yet implemented evaluation for cfg.env_suite={cfg.env_suite} environment with MuJoCo simulator during pre-training. Coming soon!")
            log.info(
                "Pretrain eval threshold: best_reward_threshold_for_success=%s",
                best_reward_threshold_for_success,
            )
            self.env_config = EnvConfig(
                n_envs=n_eval_envs,
                name=cfg.env,
                max_episode_steps=env_max_episode_steps,
                reset_at_iteration=False,
                save_video=True,                               # Change to True if needed
                use_image_obs = use_image_obs,
                best_reward_threshold_for_success=best_reward_threshold_for_success,
                wrappers=wrappers,
                n_steps=rollout_n_steps,
                render=True,
                render_num=2,
                robomimic_env_cfg_path=robomimic_env_cfg_path,
                shape_meta=shape_meta
            )
            # create mujoco simulator env
            self.venv = make_async(
                self.env_name,
                env_type=env_type,
                num_envs=self.env_config.n_envs,
                asynchronous=True,
                max_episode_steps=self.env_config.max_episode_steps,
                wrappers=self.env_config.wrappers,
                robomimic_env_cfg_path=self.env_config.robomimic_env_cfg_path,
                shape_meta=self.env_config.shape_meta,
                use_image_obs=self.env_config.use_image_obs,
                render=self.env_config.render,
                render_offscreen=self.env_config.save_video,
                obs_dim=cfg.obs_dim,
                action_dim=cfg.action_dim,
                **cfg.eval_env.specific if "specific" in cfg.env else {},
            )
            if not env_type == "furniture":
                self.venv.seed(
                    [self.seed + i for i in range(self.env_config.n_envs)]
                )
            self.n_envs = self.env_config.n_envs
            self.n_cond_step = cfg.cond_steps
            self.obs_dim = cfg.obs_dim
            self.action_dim = cfg.action_dim
            self.act_steps = cfg.horizon_steps # act_steps
            self.horizon_steps = cfg.horizon_steps
            self.max_episode_steps = self.env_config.max_episode_steps
            self.reset_at_iteration = self.env_config.reset_at_iteration
            self.furniture_sparse_reward= (
                self.env_config.specific.get("sparse_reward", False)
                if "specific" in cfg.env
                else False
            )
            # Now, replace references to cfg and its parameters with eval_config
            self.n_steps = self.env_config.n_steps
            self.best_reward_threshold_for_success = self.env_config.best_reward_threshold_for_success
            # rendering
            self.n_render = self.env_config.render_num
            self.render_video = self.env_config.save_video  # Assuming you want to use the save_video from self.env_config
            assert self.n_render <= self.n_envs, "n_render must be <= n_envs"
            assert not (
                self.n_render <= 0 and self.render_video
            ), "Need to set n_render > 0 if saving video"

        self.print_architecture()
    
    def print_architecture(self):
        import os
        arc_path=os.path.join(self.logdir, 'architecture.log')
        with open(arc_path, mode='w') as arc_file:
            arc_file.write(f"self.model=\n{self.model}\nnumber of parameters: {sum([p.numel() for p in self.model.parameters()])/1e6:.2f} M")
        log.info(f"architecture wrote to file {arc_path}")
        arc_file.close()

    # for debugging
    def test_lr_scheduler(self):
        lrs=[]
        steps=[]
        for epoch in range(self.first_epoch, self.first_epoch + self.n_epochs):
            self.lr_scheduler.step()
            lrs.append(self.optimizer.param_groups[0]["lr"])
            steps.append(epoch)
        import matplotlib.pyplot as plt 
        plt.plot(steps, lrs)
        figpath = f"agent/pretrain/test_resume.png"
        plt.savefig(figpath)
        print(f"saved lr resume test result to {figpath}")
        exit()

    def get_loss(self, batch_data):
        '''for training and validation on fixed dataset'''
        raise NotImplementedError

    def inference(self, cond:dict):
        '''for evaluation in sim'''
        raise NotImplementedError
    
    def run(self):
        print(f"dataloader_train={len(self.dataloader_train)}")
        self.test() # see the initialization or resumed performance.
        if self.only_test:
            exit()
        
        # this is for diffusion and reflow models. 
        timer = Timer()
        cnt_batch = 0
        log.info(f"self.epoch={self.epoch}, begin training.")
        # 外层循环：遍历所有训练周期(epoch)
        for epoch in tqdm(range(self.first_epoch, self.first_epoch + self.n_epochs)):
            # 设置模型为训练模式
            self.model.train()
            # train
            # 初始化用于存储本轮次训练损失的列表
            loss_train_epoch = []
            # 计算每个epoch中的训练步数（批次数）
            steps_per_epoch = len(self.dataloader_train)
            # 内层循环：遍历当前epoch中的所有训练批次
            # step: 当前批次的索引 (0, 1, 2, ...)
            # batch_train: 当前批次的数据 (包含训练所需的观测值、动作等)
            for step, batch_train in tqdm(enumerate(self.dataloader_train), desc=f'total steps={steps_per_epoch}') \
                if self.verbose_train else enumerate(self.dataloader_train):
                # 如果数据在CPU上，将其转移到指定设备
                if self.dataset_train.device == "cpu":
                    batch_train = batch_to_device(batch_train)
                
                # 清零优化器的梯度
                self.optimizer.zero_grad()
                
                # 计算当前批次的训练损失（前向传播）
                loss_train = self.get_loss(batch_train)
                
                # 反向传播计算梯度
                loss_train.backward()
                # 将当前批次的损失值添加到列表中
                loss_train_epoch.append(loss_train.item())
                # 如果启用详细损失输出，则打印当前训练进度
                if self.verbose_loss: 
                    print(f"epoch: {epoch}/{self.first_epoch + self.n_epochs}={epoch/(self.n_epochs-self.first_epoch)*100:2.2f}%, steps: {step}, loss: {loss_train.item():3.4}", end="\r")

                # 执行优化器步骤，更新模型参数
                self.optimizer.step()
                # 如果设置了每个梯度步骤都调整学习率，则更新学习率
                if self.schedule_lr_each_grad_step:
                    self.lr_scheduler.step()
                
                # update ema
                # 更新指数移动平均模型（EMA）
                if cnt_batch % self.update_ema_freq == 0:
                    self.step_ema()
                # 更新全局批次计数器
                cnt_batch += 1
            # 计算本轮次的平均训练损失
            loss_train = np.mean(loss_train_epoch)

            # validate
            # 验证阶段（如果有验证数据集且到达验证频率）
            with torch.no_grad():
                loss_val_epoch = []
                # for RL, self.dataloader_val is None. So you just skip this part. 
                # 检查是否有验证数据集且是否到达验证周期
                if self.dataloader_val is not None and self.epoch % self.val_freq == 0:
                    # 设置模型为评估模式
                    self.model.eval()
                    # 遍历验证数据集
                    for batch_val in self.dataloader_val:
                        if self.dataset_val.device == "cpu":
                            batch_val = batch_to_device(batch_val)
                        with torch.no_grad:
                            # 计算验证损失
                            loss_val = self.get_loss(batch_val)
                            loss_val_epoch.append(loss_val.item())
                    # 重新设置模型为训练模式
                    self.model.train()
                # 计算平均验证损失
                loss_val = np.mean(loss_val_epoch) if len(loss_val_epoch) > 0 else None

            # update lr
            # 更新学习率（如果不是每个梯度步骤更新）
            if not self.schedule_lr_each_grad_step:
                self.lr_scheduler.step()
            
            # always save the last checkpoint for resume 
            # 保存最新的检查点（用于恢复训练）
            self.save_last_model()
            
            # save model # default is 100 by pre_diffusion_mlp.yaml
            # 定期保存模型检查点
            if self.epoch % self.save_model_freq == 0 or self.epoch == self.n_epochs:
                self.save_model()
                        
            # test in mujoco simulator
            # 在MuJoCo模拟器中测试模型性能
            if self.test_in_mujoco and self.epoch % self.test_freq == 0:
                self.test()
            
            # log testing info
            # 记录训练日志信息
            self.log(epoch, loss_train, loss_val, timer)
            # 增加epoch计数器
            self.epoch += 1
    

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.epoch < self.epoch_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)
    
    def save_model(self):
        """
        saves model and ema to disk;
        """
        data = {
            "epoch": self.epoch,
            "model": self.model.state_dict(),
            "ema": self.ema_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
        }
        savepath = os.path.join(self.checkpoint_dir, f"state_{self.epoch}.pt")
        torch.save(data, savepath)
        log.info(f"Saved model to {savepath}\n")

    def save_best_model(self):
        data = {
            "epoch": self.epoch,
            "model": self.model.state_dict(),
            "ema": self.ema_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
        }
        savepath = os.path.join(self.checkpoint_dir, f"best.pt")
        torch.save(data, savepath)
        log.info(f"Saved the best model to {savepath}\t It has highest self.avg_episode_reward: {self.best_episode_reward:8.2f}.")

    def save_best_ema_model(self):
        data = {
            "epoch": self.epoch,
            "model": self.model.state_dict(),
            "ema": self.ema_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
        }
        savepath = os.path.join(self.checkpoint_dir, f"best_ema.pt")
        torch.save(data, savepath)
        log.info(f"Saved the best EMA model to {savepath}, which has highest self.avg_episode_reward: {self.best_episode_reward:8.2f}.")
        
    def save_last_model(self):
        '''for resume purpose'''
        data = {
            "epoch": self.epoch,
            "model": self.model.state_dict(),
            "ema": self.ema_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
        }
        savepath = os.path.join(self.checkpoint_dir, f"last.pt")
        torch.save(data, savepath)
        # log.info(f"Saved the last model to {savepath}")
    
    def load(self, epoch, custom_path=None):
        """
        loads model and ema from disk
        """
        if custom_path:
            loadpath = os.path.join(custom_path)
        else:
            loadpath = os.path.join(self.checkpoint_dir, f"state_{epoch}.pt")
        
        data = torch.load(loadpath, weights_only=True)
        
        # when resume from checkpoint, just reduce the train.n_epochs in your configuration file by the amount of epochs you have trained. 
        # our code will automatically restore the checkpoint for the model and the optimizer and the learning rate scheduler and train for another `train.n_epochs` steps. 
        self.epoch = data["epoch"]
        self.first_epoch = self.epoch+1
        log.info(f"Resume from self.epoch={self.epoch}. Will start from self.first_epoch={self.first_epoch} and train for another {self.n_epochs} epochs. ")
        self.model.load_state_dict(data["model"])
        self.ema_model.load_state_dict(data["ema"])
        self.optimizer.load_state_dict(data["optimizer"])
        self.lr_scheduler.load_state_dict(data["lr_scheduler"])
    
    # log loss
    def log(self, epoch, loss_train, loss_val, timer):
        if self.epoch % self.log_freq == 0:  # default is every step. 
            log.info(f"{self.epoch}: train loss {loss_train:8.4f} | t:{timer():8.4f}")
            if self.use_wandb:
                if loss_val is not None:
                    wandb.log(
                        {"loss - val": loss_val}, step=self.epoch, commit=False
                    )
                if not self.test_in_mujoco:
                    wandb.log(
                                {
                                "loss - train": loss_train,
                                "lr": self.optimizer.param_groups[0]["lr"],
                                },
                                step=self.epoch,
                                commit=True,
                            )
                else:
                    if self.log_all:
                        wandb.log(
                                {
                                "loss - train": loss_train,
                                "lr": self.optimizer.param_groups[0]["lr"],
                                "best_reward": self.best_episode_reward,
                                "success_rate": self.success_rate,
                                "avg_episode_reward": self.avg_episode_reward,
                                "avg_episode_reward_std": self.avg_episode_reward_std,
                                "avg_episode_length": self.avg_episode_length,
                                "avg_best_reward": self.avg_best_reward,
                                "avg_best_reward_std": self.avg_best_reward_std,
                                },
                                step=self.epoch,
                                commit=True,
                            )
                    else:
                        wandb.log(
                                {
                                "loss - train": loss_train,
                                "lr": self.optimizer.param_groups[0]["lr"],
                                "best_reward": self.best_episode_reward,
                                # "success_rate": self.success_rate,
                                # "avg_episode_reward": self.avg_episode_reward,
                                # "avg_episode_reward_std": self.avg_episode_reward_std
                                },
                                step=self.epoch,
                                commit=True,
                            )
            # TensorBoard logging (mirror of wandb but safe if writer is None)
            try:
                if getattr(self, 'writer', None) is not None:
                    # 添加每步的损失和学习率记录
                    # 注意：这里我们需要访问训练循环中的cnt_batch变量，但在这个函数作用域中不可用
                    # 因此我们只记录每个epoch级别的信息
                    
                    if loss_val is not None:
                        self.writer.add_scalar('loss/val', float(loss_val), self.epoch)
                    # train loss and lr
                    self.writer.add_scalar('loss/train', float(loss_train), self.epoch)
                    self.writer.add_scalar('lr', float(self.optimizer.param_groups[0]['lr']), self.epoch)
                    if self.test_in_mujoco:
                        if self.log_all:
                            self.writer.add_scalar('eval/best_reward', float(self.best_episode_reward), self.epoch)
                            self.writer.add_scalar('eval/success_rate', float(self.success_rate), self.epoch)
                            self.writer.add_scalar('eval/avg_episode_reward', float(self.avg_episode_reward), self.epoch)
                            self.writer.add_scalar('eval/avg_episode_reward_std', float(self.avg_episode_reward_std), self.epoch)
                            self.writer.add_scalar('eval/avg_episode_length', float(self.avg_episode_length), self.epoch)
                            self.writer.add_scalar('eval/avg_episode_length_std', float(self.avg_episode_length_std), self.epoch)
                            self.writer.add_scalar('eval/avg_best_reward', float(self.avg_best_reward), self.epoch)
                            self.writer.add_scalar('eval/avg_best_reward_std', float(self.avg_best_reward_std), self.epoch)
                        else:
                            self.writer.add_scalar('eval/best_reward', float(self.best_episode_reward), self.epoch)
            except Exception:
                pass
        # count    
        log.info(f"epoch={epoch} , self.first_epoch={self.first_epoch}, last epoch={self.first_epoch + self.n_epochs}, loss_train={loss_train:.4f}")



    ##################### test in mujoco simulator for openai gym with state inputs #####################
    
    def reset_env_all(self, verbose=False, options_venv=None, **kwargs):
        if options_venv is None:
            options_venv = [
                {k: v for k, v in kwargs.items()} for _ in range(self.n_envs)
            ]
        obs_venv = self.venv.reset_arg(options_list=options_venv)
        # convert to OrderedDict if obs_venv is a list of dict
        if isinstance(obs_venv, list):
            obs_venv = {
                key: np.stack([obs_venv[i][key] for i in range(self.n_envs)])
                for key in obs_venv[0].keys()
            }
        if verbose:
            for index in range(self.n_envs):
                logging.info(
                    f"<-- Reset environment {index} with options {options_venv[index]}"
                )
        return obs_venv

    def reset_env(self, env_ind, verbose=False):
        task = {}
        obs = self.venv.reset_one_arg(env_ind=env_ind, options=task)
        if verbose:
            logging.info(f"<-- Reset environment {env_ind} with task {task}")
        return obs
    
    def test(self):
        if not self.test_in_mujoco:
            return
        log.info(f"Evaluating {self.model.__class__.__name__} in environment {self.env_name} with denoising steps = {self.test_denoising_steps}")
        
        log_all= self.test_log_all
        timer = Timer()
        # Prepare video paths for each envs --- only applies for the first set of episodes if allowing reset within iteration and each iteration has multiple episodes from one env
        options_venv = [{} for _ in range(self.n_envs)]
        if self.render_video:
            for env_ind in range(self.n_render):
                options_venv[env_ind]["video_path"] = os.path.join(
                    self.render_dir, f"eval_trial-{env_ind}.mp4"
                )
        self.model.eval()
        firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
        prev_obs_venv = self.reset_env_all(options_venv=options_venv)
        firsts_trajs[0] = 1
        reward_trajs = np.zeros((self.n_steps, self.n_envs))
        info_history = []
        
        self.avg_episode_length = 0.0 
        # Collect a set of trajectories from env
        for step in tqdm(range(self.n_steps)) if self.verbose_test else range(self.n_steps):
            # Select action
            with torch.no_grad():
                cond = {
                    "state": torch.from_numpy(prev_obs_venv["state"])
                    .float()
                    .to(self.device)
                }
                
                # different models differs here.
                samples = self.inference(
                    cond=cond
                )
                
                # here, samples is a namedtuple of class `Sample(trajetories, chains)` trajectories is a single action forecast, and chains is the whole generation. 
                output_venv = (
                    samples.trajectories.cpu().numpy()
                )  # n_env x horizon x act
            action_venv = output_venv[:, : self.act_steps]

            # Apply multi-step action
            obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = (
                self.venv.step(action_venv)
            )
            info_history.append(info_venv)
            reward_trajs[step] = reward_venv
            firsts_trajs[step + 1] = terminated_venv | truncated_venv

            # update for next step
            prev_obs_venv = obs_venv

        # Summarize episode reward --- this needs to be handled differently depending on whether the environment is reset after each iteration. Only count episodes that finish within the iteration.
        episodes_start_end = []
        for env_ind in range(self.n_envs):
            env_steps = np.where(firsts_trajs[:, env_ind] == 1)[0]
            for i in range(len(env_steps) - 1):
                start = env_steps[i]
                end = env_steps[i + 1]
                if end - start > 1:
                    episodes_start_end.append((env_ind, start, end - 1))
                # 聚合并记录 FFSMEnv6dof 返回的 info 字段到 TensorBoard（跨步、跨环境）
        info_keys = [
            "distance_to_target",
            "base_offset_pos",
            "base_offset_theta",
            "ee_track_pos_error",
            "ee_track_atti_error",
            "collision_num",
        ]
        if len(episodes_start_end) > 0:
            reward_trajs_split = [
                reward_trajs[start : end + 1, env_ind]
                for env_ind, start, end in episodes_start_end
            ]
            num_episode_finished = len(reward_trajs_split)
            episode_reward = np.array(
                [np.sum(reward_traj) for reward_traj in reward_trajs_split]
            )
            if (
                self.furniture_sparse_reward
            ):  # only for furniture tasks, where reward only occurs in one env step
                episode_best_reward = episode_reward
            else:
                episode_best_reward = np.array(
                    [
                        np.max(reward_traj) / self.act_steps
                        for reward_traj in reward_trajs_split
                    ]
                )
            
            self.avg_episode_reward = np.mean(episode_reward)
            self.avg_episode_reward_std = np.std(episode_reward)
            self.success_rate = np.mean(
                episode_best_reward >= self.best_reward_threshold_for_success
            )
            if log_all:
                self.avg_best_reward = np.mean(episode_best_reward)
                self.avg_best_reward_std = np.std(episode_best_reward)
                episode_lengths = np.array([end - start + 1 for _, start, end in episodes_start_end]) * self.act_steps
                traj_length = np.mean(episode_lengths) if len(episode_lengths) > 0 else 0
                traj_std = np.std(episode_lengths) if len(episode_lengths) > 0 else 0  # Added
                self.avg_episode_length=traj_length
                self.avg_episode_length_std=traj_std
            # 新增：按episode统计info
            try:
                episode_mse_stats = {k: [] for k in info_keys if k != "collision_num"}
                episode_collision_sums = []
                episode_cumulative_rewards = []  # 新增：存储每条episode的累计奖励
                
                for env_ind, start, end in episodes_start_end:
                    episode_infos = []
                    episode_reward = 0.0  # 新增：当前episode的累计奖励
                    for step in range(start, end + 1):
                        if step < len(info_history):
                            infos = info_history[step]
                            if isinstance(infos, (list, tuple)) and env_ind < len(infos):
                                episode_infos.append(infos[env_ind])
                        # 新增：累计当前步骤的奖励
                        if step < len(reward_trajs) and env_ind < reward_trajs.shape[1]:
                            episode_reward += reward_trajs[step, env_ind]
                    
                    # 计算这个episode内各指标的MSE（除collision_num外）
                    episode_values = {k: [] for k in info_keys if k != "collision_num"}
                    episode_collision = 0
                    
                    for info in episode_infos:
                        if isinstance(info, dict):
                            for k in info_keys:
                                if k in info:
                                    if k != "collision_num":
                                        # 收集前5个指标的值用于计算MSE
                                        episode_values[k].append(float(info[k]))
                                    else:
                                        # 累计collision_num
                                        episode_collision += int(info[k])
                    
                    # 计算每个指标的MSE（均方误差）
                    for k, values in episode_values.items():
                        if len(values) > 0:
                            mse = np.mean(np.square(values))  # MSE = mean(square(values))
                            episode_mse_stats[k].append(mse)
                    # 记录这个episode的累计碰撞次数
                    episode_collision_sums.append(episode_collision)
                    # 新增：记录这个episode的累计奖励
                    episode_cumulative_rewards.append(episode_reward)
                # 记录到TensorBoard
                if getattr(self, 'writer', None) is not None:
                    # 记录前5个指标的MSE（所有episode的统计）
                    for k, mse_values in episode_mse_stats.items():
                        if mse_values:
                            mean_mse = float(np.mean(mse_values))
                            std_mse = float(np.std(mse_values)) if len(mse_values) > 1 else 0.0
                            self.writer.add_scalar(f'eval/info/{k}_mse_mean', mean_mse, self.epoch)
                            self.writer.add_scalar(f'eval/info/{k}_mse_std', std_mse, self.epoch)

                    # 记录collision_num的累计值统计
                    if episode_collision_sums:
                        mean_collision = float(np.mean(episode_collision_sums))
                        std_collision = float(np.std(episode_collision_sums)) if len(episode_collision_sums) > 1 else 0.0
                        total_collision = int(np.sum(episode_collision_sums))
                        self.writer.add_scalar(f'eval/info/collision_num_mean', mean_collision, self.epoch)
                        self.writer.add_scalar(f'eval/info/collision_num_std', std_collision, self.epoch)
                        self.writer.add_scalar(f'eval/info/collision_num_total', total_collision, self.epoch)
                    # 新增：记录累计奖励统计
                    if episode_cumulative_rewards:
                        mean_cumulative_reward = float(np.mean(episode_cumulative_rewards))
                        std_cumulative_reward = float(np.std(episode_cumulative_rewards)) if len(episode_cumulative_rewards) > 1 else 0.0
                        self.writer.add_scalar(f'eval/reward/episode_cumulative_mean', mean_cumulative_reward, self.epoch)
                        self.writer.add_scalar(f'eval/reward/episode_cumulative_std', std_cumulative_reward, self.epoch)
            except Exception as e:
                log.warning(f"Info statistics skipped due to error: {e}")
        else:
            self.avg_episode_reward = 0
            self.avg_episode_reward_std=0.0
            self.success_rate = 0
            if log_all:
                episode_reward = np.array([])
                num_episode_finished = 0
                self.avg_best_reward = 0
                self.avg_best_reward_std=0.0
                episode_lengths = np.array([end - start + 1 for _, start, end in episodes_start_end]) * self.act_steps
                traj_length = np.mean(episode_lengths) if len(episode_lengths) > 0 else 0
                traj_std = np.std(episode_lengths) if len(episode_lengths) > 0 else 0  # Added
                self.avg_episode_length=traj_length
                self.avg_episode_length_std=traj_std
            
            log.info("[WARNING] No episode completed within the iteration!")

        if self.avg_episode_reward > self.best_episode_reward:
            self.best_episode_reward = self.avg_episode_reward
            self.save_best_model()
            log.info(f"Current Best model saved at epoch {self.epoch}")

        time = timer()
        if log_all:
            log.info(
                f"eval: success rate {self.success_rate:8.4f} (thr={self.best_reward_threshold_for_success:4.2f}) | avg episode reward {self.avg_episode_reward:8.1f}±{self.avg_episode_reward_std:2.1f} | avg_episode_length {self.avg_episode_length:4.2f}±{self.avg_episode_length_std:4.2f} | num episode {num_episode_finished:4d} | avg best reward {self.avg_best_reward:8.1f}±{self.avg_best_reward_std:2.1f} |"
            )
        else:
            log.info(
                f"eval: success rate {self.success_rate*100:2.2f}% (thr={self.best_reward_threshold_for_success:4.2f})| avg episode reward {self.avg_episode_reward:8.1f}±{self.avg_episode_reward_std:2.1f}"
            )
        if log_all:
            np.savez(
                self.result_path,
                num_episode=num_episode_finished,
                eval_success_rate=self.success_rate,
                eval_episode_reward=self.avg_episode_reward,
                eval_best_reward=self.avg_best_reward,
                time=time,
            )
        else:
            np.savez(
                self.result_path,
                eval_success_rate=self.success_rate,
                eval_episode_reward=self.avg_episode_reward,
                eval_episode_reward_std=self.avg_episode_reward_std,
                time=time,
            )

    def eval(self):
        "加载模型,评估并输出gif图,固定使用1个环境进行评估"
        log.info(f"Evaluating {self.model.__class__.__name__} in environment {self.env_name} with denoising steps = {self.test_denoising_steps}")
        log_all= self.test_log_all

        # choose checkpoint to load (prefer EMA then best then last)
        ckpt = None
        candidates = []
        if self.resume_path:
            candidates.append(self.resume_path)
        if self.checkpoint_dir:
            candidates += [
                os.path.join(self.checkpoint_dir, 'best_ema.pt'),
                os.path.join(self.checkpoint_dir, 'best.pt'),
                os.path.join(self.checkpoint_dir, 'last.pt'),
            ]

        for c in candidates:
            if c and os.path.exists(c):
                ckpt = c
                break

        if ckpt is None:
            log.warning('No checkpoint found for eval; using current in-memory model')
        else:
            try:
                data = torch.load(ckpt, map_location=self.device)
                # pick ema or model according to test_model_type
                if self.test_model_type == 'ema' and 'ema' in data:
                    self.model.load_state_dict(data['ema'])
                elif 'model' in data:
                    self.model.load_state_dict(data['model'])
                else:
                    # fallback: try load whole dict
                    try:
                        self.model.load_state_dict(data)
                    except Exception:
                        log.warning('Could not load checkpoint into model')
                log.info(f'Loaded checkpoint {ckpt} for evaluation')
            except Exception as e:
                log.warning(f'Failed to load checkpoint {ckpt}: {e}')

        self.model.eval()

        # prepare save directory under checkpoint path if available (requirement from user)
        ts = datetime.datetime.now().strftime('%m%d_%H%M%S')
        if ckpt is not None:
            base_ckpt_dir = os.path.dirname(os.path.abspath(ckpt))
            base_dir = os.path.join(base_ckpt_dir, 'eval_runs', f'eval_{ts}')
        else:
            base_dir = os.path.join(self.logdir, 'eval_runs', f'eval_{ts}')
        os.makedirs(base_dir, exist_ok=True)

        # number of episodes to run (single env)
        num_episodes = getattr(self, 'eval_num', 1)

        metrics_num_colli = []
        metrics_cum_reward = []
        # local save options
        save_gif = True
        save_traj_csv = True
        save_frames = False

        # create a single env via make_async (num_envs=1)
        try:
            # prefer to reuse the same env_type used to construct the main vector env
            env_type_single = getattr(self, 'env_type', None)
            venv_single = make_async(
                self.env_name,
                env_type=env_type_single,
                num_envs=1,
                asynchronous=False,
                max_episode_steps=self.max_episode_steps,
                wrappers=self.env_config.wrappers if hasattr(self, 'env_config') else None,
                robomimic_env_cfg_path=(self.env_config.robomimic_env_cfg_path if hasattr(self, 'env_config') else None),
                shape_meta=(self.env_config.shape_meta if hasattr(self, 'env_config') else None),
                use_image_obs=(self.env_config.use_image_obs if hasattr(self, 'env_config') else False),
                render=True,
                render_offscreen=True,  # Always enable offscreen rendering for frame capture
                obs_dim=getattr(self, 'obs_dim', None),
                action_dim=getattr(self, 'action_dim', None),
            )
            log.info(f"Successfully created single environment for eval with render_offscreen=True")
        except Exception as e:
            log.exception(f'Failed to create single env via make_async. Falling back to existing venv for index 0')
            venv_single = None

        for epi in range(num_episodes):
            traj_dir = os.path.join(base_dir, f'traj_{epi:05d}')
            os.makedirs(traj_dir, exist_ok=True)

            # reset env
            env_reset_success = False
            if venv_single is not None:
                for reset_attempt in range(3):  # Try multiple reset methods
                    try:
                        obs = venv_single.reset()
                        if obs is not None:
                            log.info(f"Episode {epi}: Environment reset successful on attempt {reset_attempt + 1}")
                            env_reset_success = True
                            break
                    except Exception as e1:
                        log.warning(f"Episode {epi}: Standard reset failed on attempt {reset_attempt + 1}: {e1}")
                        try:
                            obs = venv_single.reset_arg(options_list=[{}])[0]
                            if obs is not None:
                                log.info(f"Episode {epi}: Alternative reset successful on attempt {reset_attempt + 1}")
                                env_reset_success = True
                                break
                        except Exception as e2:
                            log.warning(f"Episode {epi}: Alternative reset failed on attempt {reset_attempt + 1}: {e2}")
                            try:
                                # Try synchronous vector env reset
                                obs = venv_single.reset_wait()
                                if obs is not None:
                                    log.info(f"Episode {epi}: Synchronous reset successful on attempt {reset_attempt + 1}")
                                    env_reset_success = True
                                    break
                            except Exception as e3:
                                log.warning(f"Episode {epi}: Synchronous reset failed on attempt {reset_attempt + 1}: {e3}")
            
            if not env_reset_success:
                if venv_single is None:
                    try:
                        obs = self.reset_env(0)
                        env_reset_success = True
                        log.info(f"Episode {epi}: Using fallback environment reset")
                    except Exception as e:
                        log.error(f"Episode {epi}: All reset methods failed: {e}")
                        continue  # Skip this episode
                else:
                    log.error(f"Episode {epi}: Failed to reset environment after all attempts, skipping episode")
                    continue  # Skip this episode

            done = False
            step = 0
            frames = []
            traj_rows = []
            cum_reward = 0.0
            num_colli = 0

            # get unwrapped env for rendering/state access if possible
            unwrapped_env = None
            try:
                if venv_single is not None:
                    envs_list = getattr(venv_single, 'envs', None)
                    if envs_list is not None:
                        unwrapped_env = envs_list[0].env.unwrapped
                else:
                    envs_list = getattr(self.venv, 'envs', None)
                    if envs_list is not None:
                        unwrapped_env = envs_list[0].env.unwrapped
            except Exception:
                unwrapped_env = None

            max_steps = int(self.max_episode_steps or 1000)

            while not done and step < max_steps:
                # prepare cond dict similar to test()
                try:
                    if isinstance(obs, dict) and 'state' in obs:
                        state_np = obs['state']
                    else:
                        # obs might be array-like
                        state_np = np.array(obs)
                    cond = {
                        'state': torch.from_numpy(np.asarray(state_np)).float().to(self.device)
                    }
                    # ensure batch dim
                    if cond['state'].dim() == 1:
                        cond['state'] = cond['state'].unsqueeze(0)
                    elif cond['state'].dim() == 3:
                        # already batched as [1,...]
                        pass
                except Exception:
                    log.warning('Failed to build cond from obs; breaking')
                    break

                with torch.no_grad():
                    samples = self.inference(cond=cond)
                try:
                    output = samples.trajectories.cpu().numpy()
                except Exception:
                    log.warning('Inference did not return expected samples. Breaking.')
                    break

                action = output[0, : self.act_steps]
                action_batch = np.expand_dims(action, 0)

                # step the single env
                try:
                    if venv_single is not None:
                        obs_n, reward_n, terminated_n, truncated_n, info_n = venv_single.step(action_batch)
                        obs = obs_n
                        reward_val = float(np.asarray(reward_n).ravel()[0])
                        done = bool((terminated_n | truncated_n).ravel()[0])
                        info = info_n[0] if isinstance(info_n, (list, tuple)) else info_n
                    else:
                        # use existing vector env stepping for index 0 via step with batch and take first
                        obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = self.venv.step(action_batch)
                        obs = {k: v[0] for k, v in obs_venv.items()} if isinstance(obs_venv, dict) else obs_venv[0]
                        reward_val = float(np.asarray(reward_venv).ravel()[0])
                        done = bool((terminated_venv | truncated_venv).ravel()[0])
                        info = info_venv[0] if isinstance(info_venv, (list, tuple)) else info_venv
                except Exception as e:
                    log.warning(f'Env step failed: {e}. Ending episode.')
                    break

                cum_reward += reward_val

                # collect collision from info
                try:
                    if isinstance(info, dict):
                        collision_num = info.get('collision_num', 0)
                        num_colli += int(collision_num)
                        log.debug(f"Episode {epi}, step {step}: collision_num from info = {collision_num}")
                    elif isinstance(info, (list, tuple)) and len(info) > 0:
                        d = info[0]
                        if isinstance(d, dict):
                            collision_num = d.get('collision_num', 0)
                            num_colli += int(collision_num)
                            log.debug(f"Episode {epi}, step {step}: collision_num from info[0] = {collision_num}")
                    else:
                        log.debug(f"Episode {epi}, step {step}: info structure not as expected: {type(info)}")
                except Exception as e:
                    log.warning(f"Failed to collect collision info at step {step}: {e}")

                # capture frame if renderer available
                frame_capture_success = False
                try:
                    log.debug(f"Episode {epi}, step {step}: Attempting frame capture...")
                    
                    if unwrapped_env is not None:
                        # Try multiple rendering approaches
                        frame = None
                        render_method_used = "none"
                        
                        # Method 1: Try mujoco_renderer
                        if hasattr(unwrapped_env, 'mujoco_renderer'):
                            try:
                                log.debug(f"Episode {epi}, step {step}: Trying mujoco_renderer...")
                                renderer = unwrapped_env.mujoco_renderer
                                
                                if getattr(renderer, 'viewer', None) is None:
                                    log.debug(f"Episode {epi}, step {step}: Initializing mujoco_renderer viewer...")
                                    renderer._get_viewer(render_mode='rgb_array')
                                
                                if hasattr(renderer, 'default_cam_config'):
                                    renderer._set_cam_config()
                                
                                if hasattr(renderer, 'data'):
                                    renderer.data = unwrapped_env.data
                                
                                renderer.camera_id = 0
                                frame = renderer.render(render_mode='rgb_array')
                                
                                if frame is not None and frame.size > 0:
                                    frames.append(np.asarray(frame))
                                    frame_capture_success = True
                                    render_method_used = "mujoco_renderer"
                                    log.info(f"Episode {epi}, step {step}: Successfully captured {len(frames)} frames with mujoco_renderer, shape: {frame.shape}")
                                else:
                                    log.warning(f"Episode {epi}, step {step}: mujoco_renderer returned None or empty frame")
                            except Exception as e:
                                log.warning(f"Episode {epi}, step {step}: mujoco_renderer failed: {e}")
                        else:
                            log.debug(f"Episode {epi}, step {step}: mujoco_renderer not available")
                        
                        # Method 2: Try env.render if renderer method 1 failed or not available
                        if not frame_capture_success:
                            try:
                                log.debug(f"Episode {epi}, step {step}: Trying env.render...")
                                frame = unwrapped_env.render(mode='rgb_array')
                                
                                if frame is not None and frame.size > 0:
                                    frames.append(np.asarray(frame))
                                    frame_capture_success = True
                                    render_method_used = "env.render"
                                    log.info(f"Episode {epi}, step {step}: Successfully captured {len(frames)} frames with env.render, shape: {frame.shape}")
                                else:
                                    log.warning(f"Episode {epi}, step {step}: env.render returned None or empty frame")
                            except Exception as e:
                                log.warning(f"Episode {epi}, step {step}: env.render failed: {e}")
                        
                        # Method 3: Try venv.render if other methods failed
                        if not frame_capture_success and venv_single is not None:
                            try:
                                log.debug(f"Episode {epi}, step {step}: Trying venv.render...")
                                frames_tuple = venv_single.render(mode='rgb_array')
                                
                                if frames_tuple is not None and len(frames_tuple) > 0:
                                    frame = frames_tuple[0]  # Take first env frame
                                    if frame is not None and frame.size > 0:
                                        frames.append(np.asarray(frame))
                                        frame_capture_success = True
                                        render_method_used = "venv.render"
                                        log.info(f"Episode {epi}, step {step}: Successfully captured {len(frames)} frames with venv.render, shape: {frame.shape}")
                                    else:
                                        log.warning(f"Episode {epi}, step {step}: venv.render first frame is None or empty")
                                else:
                                    log.warning(f"Episode {epi}, step {step}: venv.render returned empty tuple")
                            except Exception as e:
                                log.warning(f"Episode {epi}, step {step}: venv.render failed: {e}")
                        
                        # Method 4: Try alternative frame methods
                        if not frame_capture_success:
                            log.debug(f"Episode {epi}, step {step}: Trying alternative frame methods...")
                            alternative_methods = ['_get_frame', 'get_frame', 'generate_frame']
                            
                            for method_name in alternative_methods:
                                try:
                                    if hasattr(unwrapped_env, method_name):
                                        log.debug(f"Episode {epi}, step {step}: Trying {method_name}...")
                                        method = getattr(unwrapped_env, method_name)
                                        frame = method()
                                        
                                        if frame is not None and frame.size > 0:
                                            frames.append(np.asarray(frame))
                                            frame_capture_success = True
                                            render_method_used = method_name
                                            log.info(f"Episode {epi}, step {step}: Successfully captured {len(frames)} frames with {method_name}, shape: {frame.shape}")
                                            break
                                        else:
                                            log.warning(f"Episode {epi}, step {step}: {method_name} returned None or empty frame")
                                    else:
                                        log.debug(f"Episode {epi}, step {step}: {method_name} not available")
                                except Exception as e:
                                    log.warning(f"Episode {epi}, step {step}: {method_name} failed: {e}")
                        
                        # Method 5: Try sim.render
                        if not frame_capture_success:
                            try:
                                log.debug(f"Episode {epi}, step {step}: Trying sim.render...")
                                if hasattr(unwrapped_env, 'sim') and hasattr(unwrapped_env.sim, 'render'):
                                    frame = unwrapped_env.sim.render(width=640, height=480, camera_name=None, mode='offscreen')
                                    
                                    if frame is not None and frame.size > 0:
                                        frames.append(np.asarray(frame))
                                        frame_capture_success = True
                                        render_method_used = "sim.render"
                                        log.info(f"Episode {epi}, step {step}: Successfully captured {len(frames)} frames with sim.render, shape: {frame.shape}")
                                    else:
                                        log.warning(f"Episode {epi}, step {step}: sim.render returned None or empty frame")
                                else:
                                    log.debug(f"Episode {epi}, step {step}: sim.render not available")
                            except Exception as e:
                                log.warning(f"Episode {epi}, step {step}: sim.render failed: {e}")
                        
                        # Method 6: Force enable OpenGL rendering for FFSM environments
                        if not frame_capture_success:
                            try:
                                log.debug(f"Episode {epi}, step {step}: Trying force OpenGL rendering...")
                                import os
                                os.environ["MUJOCO_GL"] = "glfw"
                                
                                # Try to get viewer and render
                                if hasattr(unwrapped_env, 'viewer') and unwrapped_env.viewer is None:
                                    unwrapped_env.viewer = mujoco_viewer.MujocoViewer(unwrapped_env)
                                
                                # Try direct rendering
                                frame = unwrapped_env.render(mode='rgb_array')
                                
                                if frame is not None and frame.size > 0:
                                    frames.append(np.asarray(frame))
                                    frame_capture_success = True
                                    render_method_used = "force_opengl"
                                    log.info(f"Episode {epi}, step {step}: Successfully captured {len(frames)} frames with force OpenGL, shape: {frame.shape}")
                                else:
                                    log.warning(f"Episode {epi}, step {step}: Force OpenGL rendering returned None or empty frame")
                            except Exception as e:
                                log.warning(f"Episode {epi}, step {step}: Force OpenGL rendering failed: {e}")
                        
                        # Log summary of frame capture attempt
                        if frame_capture_success:
                            log.info(f"Episode {epi}, step {step}: Frame capture successful using {render_method_used}")
                        else:
                            log.warning(f"Episode {epi}, step {step}: All frame capture methods failed")
                    else:
                        log.debug(f"Episode {epi}, step {step}: No unwrapped environment available for frame capture")
                        
                except Exception as e:
                    log.warning(f"Episode {step}: Frame capture completely failed: {e}")
                    traceback.print_exc()
                
                # If no frames captured and we have reached a reasonable step count, create dummy frames for testing
                if not frame_capture_success and len(frames) == 0 and step > 0:
                    try:
                        log.debug(f"Episode {epi}, step {step}: Creating dummy frame for testing...")
                        import PIL.Image as Image
                        import numpy as np
                        
                        # Create a simple colored frame based on step number
                        color_value = (step * 10) % 255
                        frame = np.full((480, 640, 3), color_value, dtype=np.uint8)
                        frames.append(frame)
                        log.info(f"Episode {epi}, step {step}: Created dummy frame, total frames: {len(frames)}")
                    except Exception as e:
                        log.warning(f"Episode {epi}, step {step}: Failed to create dummy frame: {e}")

                # collect per-step state metrics: try access unwrapped env
                try:
                    if unwrapped_env is not None:
                        position = unwrapped_env.data.qpos.flat.copy()
                        p_base = position[:3]
                        q_base = position[3:7]
                        p_ee = unwrapped_env.data.geom_xpos[unwrapped_env.model.geom('fringertip').id]
                        xmat_ee = unwrapped_env.data.geom_xmat[unwrapped_env.model.geom('fringertip').id].reshape(3, 3)
                        p_targ = position[14:17]
                        xmat_targ = unwrapped_env.data.geom_xmat[unwrapped_env.model.geom('target').id].reshape(3, 3)
                        p_e = p_ee - p_targ
                        # fall back: compute placeholder attitude errors (adapter can compute better if needed)
                        att_e = np.zeros(3, dtype=np.float32)
                        att_b = np.zeros(3, dtype=np.float32)
                        row = list(p_e[:3]) + list(att_e[:3]) + list(p_base[:3]) + list(att_b[:3])
                        traj_rows.append(row)
                except Exception:
                    # fallback to info dict
                    try:
                        d = info if isinstance(info, dict) else (info[0] if isinstance(info, (list, tuple)) else {})
                        p_e = d.get('ee_track_pos_error', [0.0, 0.0, 0.0])
                        att_e = d.get('ee_track_atti_error', [0.0, 0.0, 0.0])
                        p_b = d.get('base_offset_pos', [0.0, 0.0, 0.0])
                        att_b = d.get('base_offset_theta', [0.0, 0.0, 0.0])
                        row = list(p_e[:3]) + list(att_e[:3]) + list(p_b[:3]) + list(att_b[:3])
                        traj_rows.append(row)
                    except Exception:
                        pass

                step += 1

            # end episode loop
            metrics_num_colli.append(num_colli)
            metrics_cum_reward.append(cum_reward)

            # save frames
            if save_gif:
                gif_path = os.path.join(traj_dir, 'traj.gif')
                
                if len(frames) > 0:
                    log.info(f"Episode {epi}: Attempting to save GIF with {len(frames)} frames to {gif_path}")
                    
                    # Verify all frames are valid
                    valid_frames = 0
                    for i, frame in enumerate(frames):
                        if frame is not None and frame.size > 0:
                            valid_frames += 1
                            log.debug(f"Episode {epi} - Frame {i}: shape {frame.shape}, type {frame.dtype}, size {frame.size}")
                        else:
                            log.warning(f"Episode {epi} - Frame {i} is invalid: {frame}")
                    
                    log.info(f"Episode {epi}: Found {valid_frames} valid frames out of {len(frames)} total frames")
                    
                    if valid_frames > 0:
                        # Use imageio for GIF saving
                        try:
                            import imageio.v2 as imageio
                            
                            # Ensure all frames are uint8 format
                            processed_frames = []
                            for frame in frames:
                                if frame is not None and frame.size > 0:
                                    if frame.dtype != np.uint8:
                                        frame = (frame * 255).astype(np.uint8) if frame.max() <= 1 else frame.astype(np.uint8)
                                    processed_frames.append(frame)
                            
                            log.info(f"Episode {epi}: Processed {len(processed_frames)} frames for GIF")
                            
                            # Method 1: Standard imageio.mimsave
                            try:
                                log.info(f"Episode {epi}: Attempting imageio.mimsave...")
                                imageio.mimsave(gif_path, processed_frames, duration=0.1, loop=0)
                                
                                # Verify file was created and has content
                                if os.path.exists(gif_path) and os.path.getsize(gif_path) > 0:
                                    file_size = os.path.getsize(gif_path)
                                    log.info(f"Episode {epi}: Successfully saved GIF: {gif_path} with {len(processed_frames)} frames, size: {file_size} bytes")
                                else:
                                    log.error(f"Episode {epi}: GIF file was not created or is empty: {gif_path}")
                            except Exception as e:
                                log.error(f"Episode {epi}: imageio.mimsave failed: {e}")
                                import traceback
                                traceback.print_exc()
                            
                            # Method 2: Alternative save method using PIL if method 1 failed
                            if not os.path.exists(gif_path) or os.path.getsize(gif_path) == 0:
                                try:
                                    log.info(f"Episode {epi}: Attempting PIL-based save method...")
                                    from PIL import Image
                                    
                                    # Convert numpy arrays to PIL Images
                                    pil_frames = []
                                    for i, frame in enumerate(processed_frames):
                                        try:
                                            if len(frame.shape) == 3:  # RGB image
                                                img = Image.fromarray(frame)
                                            else:  # Grayscale
                                                img = Image.fromarray(frame.squeeze())
                                            pil_frames.append(img)
                                            log.debug(f"Episode {epi}: Converted frame {i} to PIL Image")
                                        except Exception as e:
                                            log.warning(f"Episode {epi}: Failed to convert frame {i} to PIL Image: {e}")
                                    
                                    if len(pil_frames) > 0:
                                        # Save as GIF
                                        pil_frames[0].save(
                                            gif_path,
                                            save_all=True,
                                            append_images=pil_frames[1:],
                                            duration=100,  # 100ms per frame
                                            loop=0
                                        )
                                        
                                        # Verify file was created and has content
                                        if os.path.exists(gif_path) and os.path.getsize(gif_path) > 0:
                                            file_size = os.path.getsize(gif_path)
                                            log.info(f"Episode {epi}: Successfully saved PIL-based GIF: {gif_path} with {len(pil_frames)} frames, size: {file_size} bytes")
                                        else:
                                            log.error(f"Episode {epi}: PIL-based GIF file was not created or is empty: {gif_path}")
                                    else:
                                        log.error(f"Episode {epi}: No valid PIL frames for GIF save")
                                except Exception as e:
                                    log.error(f"Episode {epi}: PIL-based GIF save failed: {e}")
                                    import traceback
                                    traceback.print_exc()
                            
                            # Method 3: Manual frame saving as PNG sequence if GIF fails
                            if not os.path.exists(gif_path) or os.path.getsize(gif_path) == 0:
                                try:
                                    log.info(f"Episode {epi}: Attempting to save individual frames as PNG sequence...")
                                    png_dir = os.path.join(traj_dir, 'frames')
                                    os.makedirs(png_dir, exist_ok=True)
                                    
                                    saved_frames = 0
                                    for i, frame in enumerate(processed_frames):
                                        frame_path = os.path.join(png_dir, f"frame_{i:04d}.png")
                                        
                                        try:
                                            if len(frame.shape) == 3:  # RGB
                                                img = Image.fromarray(frame)
                                            else:  # Grayscale
                                                img = Image.fromarray(frame.squeeze())
                                            img.save(frame_path)
                                            saved_frames += 1
                                            log.debug(f"Episode {epi}: Saved frame {i} to {frame_path}")
                                        except Exception as e:
                                            log.error(f"Episode {epi}: Failed to save frame {i}: {e}")
                                    
                                    if saved_frames > 0:
                                        log.info(f"Episode {epi}: Successfully saved {saved_frames} frames as PNG sequence to {png_dir}")
                                        # Try to create a simple GIF from the PNG sequence using PIL
                                        try:
                                            png_frames = []
                                            for i in range(saved_frames):
                                                frame_path = os.path.join(png_dir, f"frame_{i:04d}.png")
                                                if os.path.exists(frame_path):
                                                    png_frames.append(Image.open(frame_path))
                                            
                                            if len(png_frames) > 0:
                                                gif_path_png = os.path.join(traj_dir, 'traj_from_png.gif')
                                                png_frames[0].save(
                                                    gif_path_png,
                                                    save_all=True,
                                                    append_images=png_frames[1:],
                                                    duration=100,
                                                    loop=0
                                                )
                                                log.info(f"Episode {epi}: Created GIF from PNG sequence: {gif_path_png}")
                                        except Exception as e:
                                            log.error(f"Episode {epi}: Failed to create GIF from PNG sequence: {e}")
                                    else:
                                        log.error(f"Episode {epi}: No frames were saved as PNG")
                                except Exception as e:
                                    log.error(f"Episode {epi}: PNG sequence save failed: {e}")
                            
                            # Final status check
                            if os.path.exists(gif_path) and os.path.getsize(gif_path) > 0:
                                log.info(f"Episode {epi}: GIF save completed successfully")
                            else:
                                # Check if PNG sequence was created
                                png_dir = os.path.join(traj_dir, 'frames')
                                if os.path.exists(png_dir) and len(os.listdir(png_dir)) > 0:
                                    log.info(f"Episode {epi}: PNG sequence was created as fallback (no GIF)")
                                else:
                                    log.error(f"Episode {epi}: All save methods failed - no GIF or PNG sequence created")
                                
                        except Exception as e:
                            log.error(f"Episode {epi}: GIF save process failed: {e}")
                            import traceback
                            traceback.print_exc()
                    else:
                        log.error(f"Episode {epi}: No valid frames found for GIF save")
                else:
                    log.warning(f"Episode {epi}: No frames captured, cannot save GIF")

            if save_frames and len(frames) > 0:
                frames_out_dir = os.path.join(traj_dir, 'frames')
                os.makedirs(frames_out_dir, exist_ok=True)
                for fi, frame in enumerate(frames):
                    fname = os.path.join(frames_out_dir, f'frame_{fi:04d}.png')
                    try:
                        imageio.imsave(fname, frame)
                    except Exception:
                        pass

            if save_traj_csv and len(traj_rows) > 0:
                csv_path = os.path.join(traj_dir, 'traj.csv')
                header = 'P_e_x,P_e_y,P_e_z,atti_e_x,atti_e_y,atti_e_z,P_b_x,P_b_y,P_b_z,atti_b_x,atti_b_y,atti_b_z'
                try:
                    np.savetxt(csv_path, np.array(traj_rows), delimiter=',', fmt='%.6f', header=header, comments='')
                except Exception:
                    pass

        # end all episodes
        # save aggregated metrics with detailed information
        metrics_path = os.path.join(base_dir, 'metrics.csv')
        
        # Calculate additional statistics
        num_episodes = len(metrics_num_colli)
        avg_num_colli = np.mean(metrics_num_colli) if metrics_num_colli else 0
        avg_cum_reward = np.mean(metrics_cum_reward) if metrics_cum_reward else 0
        std_cum_reward = np.std(metrics_cum_reward) if len(metrics_cum_reward) > 1 else 0
        max_reward = np.max(metrics_cum_reward) if metrics_cum_reward else 0
        min_reward = np.min(metrics_cum_reward) if metrics_cum_reward else 0
        
        # Create comprehensive metrics
        metrics_data = {
            'episode': list(range(num_episodes)),
            'num_colli': metrics_num_colli,
            'cum_reward': metrics_cum_reward,
            'avg_num_colli': [avg_num_colli] * num_episodes,
            'avg_cum_reward': [avg_cum_reward] * num_episodes,
            'std_cum_reward': [std_cum_reward] * num_episodes,
            'max_reward': [max_reward] * num_episodes,
            'min_reward': [min_reward] * num_episodes,
        }
        
        # Save detailed metrics CSV
        try:
            import pandas as pd
            df = pd.DataFrame(metrics_data)
            df.to_csv(metrics_path, index=False, float_format='%.6f')
            log.info(f'Saved detailed eval metrics to {metrics_path}')
            log.info(f'Evaluation Summary: {num_episodes} episodes, avg reward: {avg_cum_reward:.2f}±{std_cum_reward:.2f}, avg collisions: {avg_num_colli:.2f}')
        except Exception as e:
            log.warning(f'Failed to save detailed metrics: {e}')
            # Fallback to simple format
            try:
                data_array = np.column_stack((np.array(metrics_num_colli), np.array(metrics_cum_reward)))
                np.savetxt(metrics_path, data_array, delimiter=',', fmt='%.6f', header='num_colli,cum_reward', comments='')
                log.info(f'Saved basic eval metrics to {metrics_path}')
            except Exception as e2:
                log.error(f'Failed to save basic metrics: {e2}')

        # close env if created
        try:
            if venv_single is not None:
                venv_single.close()
        except Exception:
            pass
class EMA:
    """
    Exponential moving average
    """
    def __init__(self, cfg):
        super().__init__()
        self.beta = cfg.decay
    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)
    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new
