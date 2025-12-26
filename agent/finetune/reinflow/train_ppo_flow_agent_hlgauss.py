from agent.finetune.reinflow.train_ppo_flow_agent import TrainPPOFlowAgent
import torch
import torch.nn.functional as F
import logging
import numpy as np
import os

log = logging.getLogger(__name__)

class ValueEstimator(torch.nn.Module):
    def __init__(self, critic, hl_gauss_loss):
        super().__init__()
        self.critic = critic
        self.hl_gauss_loss = hl_gauss_loss
    
    def forward(self, obs, **kwargs):
        logits = self.critic(obs, **kwargs)
        probs = F.softmax(logits, dim=-1)
        values = self.hl_gauss_loss.transform_from_probs(probs)
        return values.unsqueeze(-1) # (B, 1)

class TrainPPOFlowAgentHLGauss(TrainPPOFlowAgent):
    def get_value(self, cond:dict, device='cpu'):
        # cond contains a floating-point torch.tensor on self.device
        with torch.no_grad():
            logits = self.model.critic.forward(cond)
            probs = F.softmax(logits, dim=-1)
            value = self.model.hl_gauss_loss.transform_from_probs(probs)
            
        if device == 'cpu':
            value_venv = value.cpu().numpy().flatten()
        else:
            value_venv = value.squeeze().float().to(self.device)
        return value_venv

    def run(self):
        self.init_buffer()
        self.prepare_run()
        self.buffer.reset() # as long as we put items at the right position in the buffer (determined by 'step'), the buffer automatically resets when new iteration begins (step =0). so we only need to reset in the beginning. This works only for PPO buffer, otherwise may need to reset when new iter begins.
        if self.resume:
            self.resume_training()
        while self.itr < self.n_train_itr:
            self.prepare_video_path()
            self.set_model_mode()
            self.reset_env() # for gpu version, add device=self.device
            self.buffer.update_full_obs()
            for step in range(self.n_steps):
                
                with torch.no_grad():
                    cond = {
                        "state": torch.tensor(self.prev_obs_venv["state"], device=self.device, dtype=torch.float32)
                    }
                    value_venv = self.get_value(cond=cond) # for gpu version add , device=self.device
                    action_samples, chains_venv, logprob_venv = self.get_samples_logprobs(cond=cond, 
                                                                                          normalize_denoising_horizon=self.normalize_denoising_horizon,
                                                                                          normalize_act_space_dimension=self.normalize_act_space_dim, 
                                                                                          clip_intermediate_actions=self.clip_intermediate_actions,
                                                                                          account_for_initial_stochasticity=self.account_for_initial_stochasticity) # for gpu version, add , device=self.device
                
                # Apply multi-step action
                action_venv = action_samples[:, : self.act_steps]
                obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = self.venv.step(action_venv)
                
                self.buffer.save_full_obs(info_venv)
                self.buffer.add(step, self.prev_obs_venv["state"], chains_venv, reward_venv, terminated_venv, truncated_venv, value_venv, logprob_venv)
                
                self.prev_obs_venv = obs_venv
                self.cnt_train_step+= self.n_envs * self.act_steps if not self.eval_mode else 0
            self.buffer.summarize_episode_reward()
            
            if not self.eval_mode:
                # Use ValueEstimator to wrap critic and return scalar values
                value_estimator = ValueEstimator(self.model.critic, self.model.hl_gauss_loss)
                self.buffer.update(obs_venv, value_estimator) # for gpu version, add device=self.device
                self.agent_update(verbose=self.verbose)
            
            # self.plot_state_trajecories() #(only in D3IL)
            
            self.log()                                          # diffusion_min_sampling_std
            if getattr(self, 'writer', None) is not None:
                self.writer.flush()
            self.update_lr()
            self.adjust_finetune_schedule()# update finetune scheduler of ReFlow Policy
            self.save_model()
            self.itr += 1 
        # Close TensorBoard writer at the end of training
        self.close_writer()
