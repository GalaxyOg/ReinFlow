# MIT License

# Copyright (c) 2025 ReinFlow Authors

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""
MLP models for flow matching with learnable stochastic interpolate noise.
"""
import torch
import torch.nn as nn
import logging
import numpy as np
from copy import deepcopy
from typing import Tuple
from torch import Tensor
from model.common.mlp import MLP, ResidualMLP
from model.diffusion.modules import SinusoidalPosEmb
from model.common.modules import SpatialEmb, RandomShiftsAug
from model.common.vit import VitEncoder
log = logging.getLogger(__name__)
import einops
from typing import List

class FlowMLP(nn.Module):
    def __init__(
        self,
        horizon_steps,
        action_dim,
        cond_dim,
        time_dim=16,
        mlp_dims=[256, 256],
        cond_mlp_dims=None,
        activation_type="Mish",
        out_activation_type="Identity",
        use_layernorm=False,
        residual_style=False,
    ):
        """
        FlowMLP类初始化函数
        用于构建基于MLP的Rectified Flow模型，预测给定状态和时间步的速度场
        
        参数说明:
        - horizon_steps: int, 动作序列长度（时间步数）
        - action_dim: int, 单个动作的维度
        - cond_dim: int, 条件信息维度（如状态观测）
        - time_dim: int, 时间嵌入维度，默认为16
        - mlp_dims: list, 主MLP网络各层维度，默认为[256, 256]
        - cond_mlp_dims: list, 条件编码器MLP维度，None表示不使用条件编码器
        - activation_type: str, 激活函数类型，默认为"Mish"
        - out_activation_type: str, 输出层激活函数类型，默认为"Identity"
        - use_layernorm: bool, 是否使用层归一化，默认为False
        - residual_style: bool, 是否使用残差连接，默认为False
        """
        super().__init__()
        # 时间嵌入维度
        self.time_dim = time_dim
        # 动作序列总维度 = 动作维度 × 序列长度
        self.act_dim_total = action_dim * horizon_steps
        # 动作序列长度（时间步数）
        self.horizon_steps = horizon_steps
        # 单个动作维度
        self.action_dim=action_dim
        # 条件信息维度（如状态观测）
        self.cond_dim=cond_dim
        # 主MLP网络各层维度
        self.mlp_dims=mlp_dims
        # 激活函数类型
        self.activation_type=activation_type
        # 输出层激活函数类型
        self.out_activation_type=out_activation_type
        # 是否使用层归一化
        self.use_layernorm=use_layernorm
        # 是否使用残差连接
        self.residual_style=residual_style

        # 时间嵌入模块：将时间步转换为高维特征表示
        # 结构：正弦位置编码 → 线性变换 → Mish激活 → 线性变换
        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.Mish(),
            nn.Linear(time_dim * 2, time_dim),
        )
        
        # 根据是否使用残差连接选择模型类型
        model = ResidualMLP if residual_style else MLP
        
        # 观测编码器：将条件信息（如状态）编码为固定维度的特征表示
        if cond_mlp_dims:
            # 如果指定了条件编码器维度，则创建MLP编码器
            self.cond_mlp = MLP(
                [cond_dim] + cond_mlp_dims,  # 输入维度到隐藏层维度再到输出维度
                activation_type=activation_type,
                out_activation_type="Identity",  # 输出层使用恒等激活函数
            )
            # 条件编码后的维度为最后一层的输出维度
            self.cond_enc_dim = cond_mlp_dims[-1]
        else:
            # 如果未指定条件编码器维度，则直接使用原始条件维度
            self.cond_enc_dim = cond_dim
            
        # 计算输入维度：时间嵌入维度 + 动作序列维度 + 条件编码维度
        input_dim = time_dim + action_dim * horizon_steps + self.cond_enc_dim
        
        # 速度预测头（velocity head）：根据时间、动作和条件信息预测速度场
        self.mlp_mean = model(
            [input_dim] + mlp_dims + [self.act_dim_total],  # 网络结构：输入层 → 隐藏层 → 输出层
            activation_type=activation_type,
            out_activation_type=out_activation_type,
            use_layernorm=use_layernorm,
        )
    
    def forward(
        self,
        action,
        time,
        cond,
        output_embedding=False,
        **kwargs,
    ):
        """
        前向传播函数：根据输入的动作、时间和条件信息预测速度场
        
        参数说明:
        - action: (B, Ta, Da) 张量，B为批次大小，Ta为动作序列长度，Da为单个动作维度
        - time: (B,) 或 int，扩散时间步，范围通常在[0,1)
        - cond: 字典，包含键state/rgb等条件信息，最新的观测在末尾
                state: (B, To, Do) 状态张量，B为批次大小，To为历史长度，Do为状态维度
        - output_embedding: bool，是否输出中间嵌入表示
        
        返回值:
        - velocity: 速度场预测结果
        - vel: (B, Ta, Da) 当output_embedding==False时返回速度场
        - vel,time_emb, cond_emb: 当output_embedding==True时返回速度场和中间嵌入
        """
        # 获取输入张量的形状信息
        B, Ta, Da = action.shape  # B:批次大小, Ta:动作序列长度, Da:动作维度

        # 展平动作序列：将(B, Ta, Da)展平为(B, Ta*Da)
        action = action.view(B, -1)

        # 展平观测历史：将(B, To, Do)展平为(B, To*Do)
        state = cond["state"].view(B, -1)

        # 观测编码器：将状态信息编码为条件嵌入
        cond_emb = self.cond_mlp(state) if hasattr(self, "cond_mlp") else state
        
        # 时间编码器：将时间步编码为时间嵌入
        if isinstance(time, int) or isinstance(time, float):
            # 如果时间是标量，则扩展为与批次大小相同的张量
            time=torch.ones((B,1), device=action.device)* time
        # 使用时间嵌入模块处理时间信息
        time_emb = self.time_embedding(time.view(B, 1)).view(B, self.time_dim)
        
        # 速度预测头：拼接所有特征并预测速度场
        vel_feature = torch.cat([action, time_emb, cond_emb], dim=-1)  # 拼接动作、时间、条件特征
        vel = self.mlp_mean(vel_feature)  # 通过MLP网络预测速度
        
        # 根据参数决定返回值格式
        if output_embedding:
            # 返回速度场以及中间嵌入表示
            return vel.view(B, Ta, Da), time_emb, cond_emb
        # 仅返回速度场
        return vel.view(B, Ta, Da)

    def sample_action(self,cond:dict,inference_steps:int,clip_intermediate_actions:bool,act_range:List[float], z:Tensor=None,save_chains:bool=False):
        """
        通过积分（欧拉法）生成动作序列。可以指定初始噪声。
        当`save_chains`为True时，同时返回去噪轨迹。
        
        参数说明:
        - cond: dict，条件信息字典，包含状态等观测信息
        - inference_steps: int，推理步数，即欧拉积分的步数
        - clip_intermediate_actions: bool，是否对中间动作进行裁剪
        - act_range: List[float]，动作范围限制，如[-1, 1]
        - z: Tensor，可选的初始噪声，默认为标准正态分布
        - save_chains: bool，是否保存去噪过程中的轨迹信息
        
        返回值:
        - x_hat: (B, Ta, Da) 生成的动作序列
        - x_chain: (B, inference_steps+1, Ta, Da) 可选的去噪轨迹
        """
        # 获取批次大小和设备信息
        B = cond['state'].shape[0]
        device=cond['state'].device

        # 初始化动作序列：如果提供了初始噪声z则使用，否则生成标准正态分布噪声
        x_hat:Tensor=z if z is not None else torch.randn(B, self.horizon_steps, self.action_dim, device=device)
        
        # 如果需要保存轨迹，则初始化轨迹存储张量
        if save_chains:
            x_chain=torch.zeros((B, inference_steps+1, self.horizon_steps, self.action_dim), device=device)
            
        # 计算时间步长：dt = 1/inference_steps
        dt = (1 / inference_steps) * torch.ones_like(x_hat, device=device)
        
        # 生成时间步序列：从0到1-1/inference_steps均匀分布
        steps = torch.linspace(0, 1-1 / inference_steps, inference_steps, device=device).repeat(B, 1)
        
        # 欧拉积分过程：逐步去噪生成动作序列
        for i in range(inference_steps):
            # 当前时间步
            t = steps[:, i]
            
            # 使用模型预测当前状态下的速度场
            vt = self.forward(x_hat, t, cond)
            
            # 欧拉更新：x_{t+1} = x_t + v_t * dt
            x_hat += vt * dt
            
            # 根据参数决定是否对动作进行裁剪（限制在合法范围内）
            if clip_intermediate_actions or i == inference_steps-1: # 总是对最终输出动作进行裁剪
                x_hat = x_hat.clamp(*act_range)
                
            # 如果需要保存轨迹，则记录当前状态
            if save_chains:
                x_chain[:, i+1] = x_hat
                
        # 根据参数决定返回值格式
        if save_chains:
            # 返回生成的动作序列和完整的去噪轨迹
            return x_hat, x_chain
        # 仅返回生成的动作序列
        return x_hat
    
    
class ExploreNoiseNet(nn.Module):
    '''
    探索噪声网络：生成可学习的探索噪声，条件于时间嵌入和/或状态嵌入。
    实现形式为 \sigma(s,t) 或 \sigma(s)
    '''
    def __init__(self,
                 in_dim:int,
                 out_dim:int,
                 logprob_denoising_std_range:list, #[min_std, max_std]
                 device,
                 hidden_dims=[16], #[8]  [32],
                 activation_type='Tanh'
                 ):
        """
        ExploreNoiseNet类初始化函数
        用于生成可学习的探索噪声标准差
        
        参数说明:
        - in_dim: int, 输入维度（时间嵌入维度+条件嵌入维度）
        - out_dim: int, 输出维度（动作序列总维度）
        - logprob_denoising_std_range: list, 噪声标准差范围[min_std, max_std]
        - device: 设备信息
        - hidden_dims: list, 隐藏层维度，默认为[16]
        - activation_type: str, 激活函数类型，默认为'Tanh'
        """
        super().__init__()
        # 设备信息
        self.device = device
        
        # 对数方差MLP网络：将输入特征映射到对数方差空间
        self.mlp_logvar = MLP(
            [in_dim] + hidden_dims +[out_dim],  # 网络结构：输入层 → 隐藏层 → 输出层
            activation_type=activation_type,
            out_activation_type="Identity",  # 输出层使用恒等激活函数
        ).to(self.device)
        
        # 设置噪声范围
        self.set_noise_range(logprob_denoising_std_range)
    
    def set_noise_range(self, logprob_denoising_std_range:list):
        """
        设置噪声范围参数
        
        参数说明:
        - logprob_denoising_std_range: list, 噪声标准差范围[min_std, max_std]
        """
        self.logprob_denoising_std_range=logprob_denoising_std_range
        min_logprob_denoising_std = self.logprob_denoising_std_range[0]
        max_logprob_denoising_std = self.logprob_denoising_std_range[1]
        # 计算对数方差的最小值和最大值，并设置为不可训练的参数
        self.logvar_min = torch.nn.Parameter(torch.log(torch.tensor(min_logprob_denoising_std**2, dtype=torch.float32, device=self.device)), requires_grad=False)
        self.logvar_max = torch.nn.Parameter(torch.log(torch.tensor(max_logprob_denoising_std**2, dtype=torch.float32, device=self.device)), requires_grad=False)
    
    def forward(self, noise_feature:torch.Tensor):
        """
        前向传播函数：根据输入特征生成噪声标准差
        
        参数说明:
        - noise_feature: torch.Tensor, 输入特征张量
        
        返回值:
        - noise_std: torch.Tensor, 噪声标准差
        """
        # 通过MLP网络计算对数方差
        noise_logvar    = self.mlp_logvar(noise_feature)
        # 处理噪声，将对数方差转换为标准差
        noise_std       = self.process_noise(noise_logvar)
        return noise_std
  
    def process_noise(self, noise_logvar):
        """
        处理噪声：将对数方差转换为标准差，并限制在指定范围内
        
        参数说明:
        - noise_logvar: torch.Tensor([B, Ta , Da]) 对数方差 log \sigma^2 
        
        返回值:
        - noise_std: torch.Tensor([B, 1, Ta * Da]) 标准差 sigma，浮点数值，限制在[min_logprob_denoising_std, max_logprob_denoising_std]范围内
        """
        # 保持原始对数方差值
        noise_logvar = noise_logvar
        # 使用tanh函数将对数方差压缩到[-1, 1]范围
        noise_logvar = torch.tanh(noise_logvar)
        # 将压缩后的值线性映射到[logvar_min, logvar_max]范围
        noise_logvar = self.logvar_min + (self.logvar_max - self.logvar_min) * (noise_logvar + 1)/2.0
        # 通过对数运算将对数方差转换为标准差: sigma = exp(0.5 * log_sigma^2)
        noise_std = torch.exp(0.5 * noise_logvar)
        return noise_std


class NoisyFlowMLP(nn.Module):
    def __init__(
        self,
        policy:FlowMLP,
        denoising_steps,
        learn_explore_noise_from,
        inital_noise_scheduler_type,
        min_logprob_denoising_std,
        max_logprob_denoising_std,
        learn_explore_time_embedding,
        time_dim_explore,
        use_time_independent_noise,
        device,
        noise_hidden_dims=None,
        activation_type='Tanh'
    ):  
        """
        带噪声的FlowMLP模型：在基础FlowMLP基础上增加可学习的探索噪声
        
        参数说明:
        - policy: FlowMLP, 基础的流匹配策略网络
        - denoising_steps: int, 去噪步数
        - learn_explore_noise_from: int, 从哪一步开始学习探索噪声
        - inital_noise_scheduler_type: str, 初始噪声调度器类型
        - min_logprob_denoising_std: float, 最小对数概率去噪标准差
        - max_logprob_denoising_std: float, 最大对数概率去噪标准差
        - learn_explore_time_embedding: bool, 是否学习探索时间嵌入
        - time_dim_explore: int, 探索时间嵌入维度
        - use_time_independent_noise: bool, 是否使用时间无关噪声
        - device: 设备信息
        - noise_hidden_dims: list, 噪声网络隐藏层维度
        - activation_type: str, 激活函数类型
        """
        super().__init__()
        # 设备信息
        self.device=device
        # 基础策略网络（FlowMLP）
        self.policy:FlowMLP = policy.to(self.device)
        """
        输入:  [batchsize, time_dim + cond_enc_dim]
        输出: 正张量，形状为 [batchsize, self.denoising_steps, self.horizon_steps x self.act_dim]
        """
        
        # 去噪步数
        self.denoising_steps: int = denoising_steps
        # 从哪一步开始学习探索噪声
        self.learn_explore_noise_from: int = learn_explore_noise_from
        # 初始噪声调度器类型
        self.initial_noise_scheduler_type: str = inital_noise_scheduler_type
        
        # 检查噪声标准差范围的有效性
        if min_logprob_denoising_std > max_logprob_denoising_std:
            raise ValueError(f"min_logprob_denoising_std must not exceed max_logprob_denoising_std, but received min_logprob_denoising_std={min_logprob_denoising_std} > max_logprob_denoising_std={max_logprob_denoising_std}. Revise your configuration file!")
        
        # 最小和最大对数概率去噪标准差
        self.min_logprob_denoising_std: float = min_logprob_denoising_std
        self.max_logprob_denoising_std: float = max_logprob_denoising_std    
        # 是否学习探索时间嵌入
        self.learn_explore_time_embedding: bool  = learn_explore_time_embedding
        
        # 设置对数概率噪声级别
        self.set_logprob_noise_levels()
        
        # 噪声网络隐藏层维度
        self.noise_hidden_dims=noise_hidden_dims
        # 是否使用时间无关噪声
        self.use_time_independent_noise = use_time_independent_noise
        # 探索时间嵌入维度
        self.time_dim_explore =time_dim_explore
        # 噪声激活函数类型
        self.noise_activation_type=activation_type
        
        # 初始化探索噪声网络
        self.init_exploration_noise_net()
        
    def init_exploration_noise_net(self):
        """
        初始化探索噪声网络：根据配置参数创建相应的探索噪声网络
        """
        # 根据是否使用时间无关噪声确定噪声网络输入维度
        if self.use_time_independent_noise:
            # 使用时间无关噪声：输入仅为条件编码维度
            noise_input_dim = self.policy.cond_enc_dim
            # 如果未指定噪声隐藏层维度，则使用默认值
            if not self.noise_hidden_dims:
                self.noise_hidden_dims = [16]
        else:
            # 使用时间相关噪声
            if self.learn_explore_time_embedding:
                # 学习探索时间嵌入：输入为探索时间嵌入维度+条件编码维度
                noise_input_dim = self.time_dim_explore + self.policy.cond_enc_dim
                # 创建探索时间嵌入层
                self.time_embedding_explore = nn.Embedding(num_embeddings=self.denoising_steps, 
                                                       embedding_dim = self.time_dim_explore, 
                                                       device=self.device)
            else:
                # 不学习探索时间嵌入：输入为策略网络时间维度+条件编码维度
                noise_input_dim = self.policy.time_dim + self.policy.cond_enc_dim
                # 如果未指定噪声隐藏层维度，则根据输入输出维度自动计算
                if not self.noise_hidden_dims:
                    self.noise_hidden_dims = [int(np.sqrt(noise_input_dim**2 + self.policy.act_dim_total**2))]
        
        # 创建探索噪声网络实例
        self.explore_noise_net=ExploreNoiseNet(in_dim=noise_input_dim, 
                                                out_dim=self.policy.act_dim_total,
                                                logprob_denoising_std_range=[self.min_logprob_denoising_std, self.max_logprob_denoising_std], 
                                                device=self.device,
                                                hidden_dims=self.noise_hidden_dims,
                                                activation_type=self.noise_activation_type)
    def forward(
        self,
        action,
        time,
        cond,
        learn_exploration_noise=False,
        step=-1,
        verbose=False,
        **kwargs,
    )->Tuple[Tensor, Tensor]:
        """
        前向传播函数：同时返回速度场预测和探索噪声标准差
        
        输入参数:
            x: (B, Ta, Da) 动作序列张量，B为批次大小，Ta为动作序列长度，Da为动作维度
            time: (B,) 浮点数，范围在[0,1)之间的流匹配时间
            cond: 字典，包含键state/rgb等条件信息，最新的观测在末尾
                state: (B, To, Do) 状态张量，B为批次大小，To为历史长度，Do为状态维度
            step: (B,) torch.tensor，可选，流匹配推理步数，范围从0到denoising_steps-1
            *这里，B是环境数量(n_envs)
            
        输出:
             vel: (B, Ta, Da) 速度场预测
             noise_std: (B, Ta x Da) 噪声标准差
        """
        # 获取批次大小
        B = action.shape[0]
        # 通过策略网络获取速度场预测和中间嵌入表示
        vel, time_emb, cond_emb = self.policy.forward(action, time, cond, output_embedding=True)
        
        # 噪声头（用于探索）：允许梯度流动
        if self.initial_noise_scheduler_type=='const' or step < self.learn_explore_noise_from:
            # 使用预设的噪声级别或在学习开始步数之前，使用固定的噪声级别
            noise_std = self.logprob_noise_levels[:, step].repeat(B,1)
        else:
            # 在学习开始步数之后，使用可学习的噪声网络
            if self.use_time_independent_noise:
                # 使用时间无关噪声：仅使用条件嵌入作为特征
                noise_feature = cond_emb
            else:
                # 使用时间相关噪声
                if self.learn_explore_time_embedding:
                    # 学习探索时间嵌入：使用学习的时间嵌入和条件嵌入
                    step_ts = torch.tensor(step, device = self.device).repeat(B)
                    time_emb_explore = self.time_embedding_explore(step_ts)
                    noise_feature = torch.cat([time_emb_explore, cond_emb], dim=-1)
                else:
                    # 不学习探索时间嵌入：使用策略网络的时间嵌入（detach防止梯度回传）和条件嵌入
                    noise_feature = torch.cat([time_emb.detach(), cond_emb], dim=-1)
            
            # 通过探索噪声网络生成噪声标准差
            noise_std = self.explore_noise_net.forward(noise_feature=noise_feature)
            
            # 如果启用详细日志，则打印噪声信息
            if verbose:
                log.info(f"step={step}, learnable noise = {noise_std.mean()}")
        
        # 如果启用详细日志，则打印学习状态信息
        if verbose:
            log.info(f"step={step}, set to learn from {self.learn_explore_noise_from}, will learn exploration noise ? {step >= self.learn_explore_noise_from}, noise_std={noise_std.mean()}require_grad={noise_std.requires_grad}")
        
        # 根据参数决定是否允许噪声梯度回传
        return vel, noise_std if learn_exploration_noise else noise_std.detach()

    @torch.no_grad()
    def stochastic_interpolate(self,t):
        """
        随机插值函数：根据不同类型的噪声调度器生成相应的标准差
        
        参数说明:
        - t: torch.Tensor, 时间步张量
        
        返回值:
        - std: torch.Tensor, 计算得到的标准差
        """
        # 有效的噪声调度器类型
        valid_noise_schedulers=['vp', 'lin', 'const', 'const_schedule_itr', 'learn_decay']
        
        # 根据不同的噪声调度器类型计算标准差
        if self.initial_noise_scheduler_type == 'vp':
            # VP (Variance Preserving) 调度器
            a = 0.2 #2.0
            std = torch.sqrt(a * t * (1 - t))
        elif self.initial_noise_scheduler_type == 'lin':
            # 线性调度器
            k=0.1
            b=0.0
            std = k*t+b
        elif self.initial_noise_scheduler_type == 'const' or 'const_schedule_itr':
            # 常数调度器
            std = torch.ones_like(t) * self.min_logprob_denoising_std
        else:
            # 无效的调度器类型
            raise ValueError(f"Invalid noise scheduler type {self.initial_noise_scheduler_type}, must be in the following: {valid_noise_schedulers}")
        return std
    
    @torch.no_grad()
    def set_logprob_noise_levels(self, force_level=None, verbose=False):
        '''
        创建用于对数概率计算的噪声标准差。
        生成一个形状为`[1, self.denoising_steps, self.policy.horizon_steps x self.policy.act_dim]`的张量`self.logprob_noise_levels`
        
        参数说明:
        - force_level: 强制设置的噪声级别
        - verbose: 是否输出详细信息
        '''
        # 初始化噪声级别张量
        self.logprob_noise_levels = torch.zeros(self.denoising_steps, device=self.device, requires_grad=False)
        
        # 生成时间步序列
        steps = torch.linspace(0, 1-1 /self.denoising_steps, self.denoising_steps, device=self.device)
        # 遍历每个时间步，计算对应的噪声标准差
        for i, t in enumerate(steps):
            if force_level:
                # 如果指定了强制噪声级别，则使用该值
                self.logprob_noise_levels[i] = torch.tensor(force_level, device=self.device)
            else:
                # 否则使用随机插值函数计算噪声标准差
                self.logprob_noise_levels[i] = self.stochastic_interpolate(t)
        
        # 将噪声标准差限制在指定范围内
        self.logprob_noise_levels = self.logprob_noise_levels.clamp(min=self.min_logprob_denoising_std, max=self.max_logprob_denoising_std)
        
        # 调整张量形状以匹配动作序列维度
        self.logprob_noise_levels = self.logprob_noise_levels.unsqueeze(0).unsqueeze(-1).repeat(1, 1, self.policy.horizon_steps *  self.policy.action_dim)
        
        # 如果启用详细输出，则打印噪声级别信息
        if verbose:
            log.info(f"Set logprob noise levels. self.logprob_noise_levels={self.logprob_noise_levels}")

class VisionFlowMLP(nn.Module):
    """带有ViT主干的视觉增强FlowMLP"""
    def __init__(
        self,
        backbone: VitEncoder,
        action_dim,
        horizon_steps,
        cond_dim,                       # 仅本体感知
        img_cond_steps=1,
        time_dim=16,
        mlp_dims=[256, 256],
        activation_type="Mish",
        out_activation_type="Identity",
        use_layernorm=False,
        residual_style=False,
        spatial_emb=0,
        visual_feature_dim=128,         # 视觉特征维度
        dropout=0,
        num_img=1,                      # 目前仅支持1或2
        augment=False,
    ):
        """
        VisionFlowMLP类初始化函数
        用于构建基于视觉输入的Rectified Flow模型，使用ViT作为视觉主干网络
        
        参数说明:
        - backbone: VitEncoder, ViT视觉主干网络
        - action_dim: int, 动作维度
        - horizon_steps: int, 动作序列长度
        - cond_dim: int, 条件维度（仅本体感知）
        - img_cond_steps: int, 图像条件步数，默认为1
        - time_dim: int, 时间嵌入维度，默认为16
        - mlp_dims: list, MLP网络维度，默认为[256, 256]
        - activation_type: str, 激活函数类型，默认为"Mish"
        - out_activation_type: str, 输出激活函数类型，默认为"Identity"
        - use_layernorm: bool, 是否使用层归一化，默认为False
        - residual_style: bool, 是否使用残差连接，默认为False
        - spatial_emb: int, 空间嵌入维度，默认为0
        - visual_feature_dim: int, 视觉特征维度，默认为128
        - dropout: float, Dropout比率，默认为0
        - num_img: int, 图像数量，目前仅支持1或2，默认为1
        - augment: bool, 是否使用数据增强，默认为False
        """
        super().__init__()
        
        # 动作块参数
        self.action_dim = action_dim  # 单个动作维度
        self.horizon_steps = horizon_steps  # 动作序列长度
        self.act_dim_total = action_dim * horizon_steps  # 动作序列总维度
        
        # 历史本体感知和视觉输入参数
        self.prop_dim = cond_dim    # 本体感知维度
        self.img_cond_steps = img_cond_steps  # 图像条件步数
        
        # 时间参数
        self.time_dim = time_dim  # 时间嵌入维度
        
        # 网络参数
        self.backbone = backbone  # ViT主干网络
        self.mlp_dims = mlp_dims  # MLP网络维度
        self.activation_type = activation_type  # 激活函数类型
        self.out_activation_type = out_activation_type  # 输出激活函数类型
        self.use_layernorm = use_layernorm  # 是否使用层归一化
        self.residual_style = residual_style  # 是否使用残差连接
        self.spatial_emb = spatial_emb  # 空间嵌入维度
        
        # 其他参数
        self.dropout = dropout  # Dropout比率
        self.num_img = num_img  # 图像数量
        self.augment = augment  # 是否使用数据增强
        
        # 视觉处理模块
        self.backbone = backbone
        # 如果启用数据增强，则创建随机位移增强模块
        if augment:
            self.aug = RandomShiftsAug(pad=4)
            
        # 空间嵌入处理
        if spatial_emb > 0:
            # 空间嵌入维度必须大于1
            assert spatial_emb > 1, "this is the dimension"
            if num_img == 2:
                # 双图像输入情况：创建两个压缩模块
                self.compress1 = SpatialEmb(
                    num_patch=self.backbone.num_patch,
                    patch_dim=self.backbone.patch_repr_dim,
                    prop_dim=cond_dim,
                    proj_dim=spatial_emb,
                    dropout=dropout,
                )
                self.compress2 = deepcopy(self.compress1)
            elif num_img == 1:  # 单图像输入情况
                self.compress = SpatialEmb(
                    num_patch=self.backbone.num_patch,
                    patch_dim=self.backbone.patch_repr_dim,
                    prop_dim=cond_dim,
                    proj_dim=spatial_emb,
                    dropout=dropout,
                )
            else:
                # 不支持的图像数量
                raise NotImplementedError(f"num_img={num_img} Currently we only support 1 or 2 image inputs")
            # 计算视觉特征维度
            visual_feature_dim = spatial_emb * num_img
        else: 
            # 未指定空间嵌入，使用默认的线性压缩模块
            self.compress = nn.Sequential(
                nn.Linear(self.backbone.repr_dim, visual_feature_dim),
                nn.LayerNorm(visual_feature_dim),
                nn.Dropout(dropout),
                nn.ReLU(),
            )
        # 条件编码维度 = 视觉特征维度 + 本体感知维度
        self.cond_enc_dim = visual_feature_dim + self.prop_dim     # rgb and  proprioception      
        
        # 时间嵌入模块：与FlowMLP相同结构
        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.Mish(),
            nn.Linear(time_dim * 2, time_dim),
        )
        
        # Flow网络参数计算
        input_dim = (
            time_dim + \
                action_dim * horizon_steps + \
                        self.cond_enc_dim
        )
        
        # 输出动作块维度
        output_dim = action_dim * horizon_steps
        
        # 速度预测头：与FlowMLP相同结构
        model = ResidualMLP if residual_style else MLP
        self.mlp_mean = model(
            [input_dim] + mlp_dims + [output_dim],
            activation_type=activation_type,
            out_activation_type=out_activation_type,
            use_layernorm=use_layernorm,
        )
    
    def forward(
        self,
        action,
        time,
        cond: dict,
        output_embedding=False,
        **kwargs,
    ):
        """
        前向传播函数：处理视觉和本体感知输入，预测速度场
        
        输入参数:
            action: (B, Ta, Da) 动作块，B为批次大小，Ta为动作序列长度，Da为动作维度
            time: (B,) 或 float，范围在[0,1)内的流时间
            cond: 字典，包含键state/rgb；最新的观测在末尾
                state: (B, To, Do) 状态张量
                rgb: (B, To, C, H, W) RGB图像张量
        输出:

        TODO 长期目标：更灵活地处理条件信息
        """
        # 获取输入张量形状
        B, Ta, Da = action.shape  # B:批次大小, Ta:动作序列长度, Da:动作维度
        _, T_rgb, C, H, W = cond["rgb"].shape  # T_rgb:图像序列长度, C:通道数, H:高度, W:宽度
        
        # 展平动作块
        action = action.view(B, -1)

        # 展平历史信息（本体感知，这里我们使用原始输入而无需编码）
        state = cond["state"].view(B, -1)

        # 获取最近的图像 --- 有时我们希望使用的img_cond_steps少于cond_steps（例如，1张图像但3个本体感知）
        rgb = cond["rgb"][:, -self.img_cond_steps :]
        
        # 通过通道连接条件中的图像
        if self.num_img >1:
            # 多图像情况：重新排列图像张量
            rgb = rgb.reshape(B, T_rgb, self.num_img, 3, H, W)
            rgb = einops.rearrange(rgb, "b t n c h w -> b n (t c) h w")
        elif self.num_img==1:
            # 单图像情况：重新排列图像张量
            rgb = einops.rearrange(rgb, "b t c h w -> b (t c) h w")
        else:
            # 图像数量错误
            raise ValueError(f"self.num_img={self.num_img} <1. ")
            
        # 转换RGB为float32以进行数据增强
        rgb = rgb.float()
        
        # 视觉和本体感知嵌入：获取vit输出 - 分别传入两张图像
        if self.num_img ==2:  # 双图像输入情况
            rgb1 = rgb[:, 0]
            rgb2 = rgb[:, 1]
            # 如果启用数据增强，则对图像进行增强处理
            if self.augment:
                rgb1 = self.aug(rgb1)
                rgb2 = self.aug(rgb2)
            # 通过ViT主干网络提取特征
            feat1 = self.backbone.forward(rgb1)
            feat1 = self.compress1.forward(feat1, state)
            
            feat2 = self.backbone.forward(rgb2)
            feat2 = self.compress2.forward(feat2, state)
            
            # 拼接两个图像的特征
            feat = torch.cat([feat1, feat2], dim=-1)
        elif self.num_img ==1:  # 单图像输入情况
            # 如果启用数据增强，则对图像进行增强处理
            if self.augment:
                rgb = self.aug(rgb)
            # 通过ViT主干网络提取特征
            feat = self.backbone.forward(rgb)
            # 特征压缩
            if isinstance(self.compress, SpatialEmb):
                feat = self.compress.forward(feat, state)
            else:
                feat = feat.flatten(1, -1)
                feat = self.compress(feat)
        else:
            # 不支持的图像数量
            raise NotImplementedError(f"num_img={self.num_img} Currently we only support 1 or 2 image inputs")
            
        # 拼接视觉和本体感知输入
        cond_encoded = torch.cat([feat, state], dim=-1)   

        # 时间嵌入
        time = time.view(B, 1)
        time_emb = self.time_embedding(time).view(B, self.time_dim)
        
        # 所有嵌入：时间、视觉-本体感知
        emb = torch.cat([action, time_emb, cond_encoded], dim=-1)

        # 速度预测头
        vel = self.mlp_mean(emb)
        
        # 根据参数决定返回值格式
        if output_embedding:
            return vel.view(B, Ta, Da), time_emb, cond_encoded
        return vel.view(B, Ta, Da)


class NoisyVisionFlowMLP(NoisyFlowMLP):
    def __init__(
            self,
            policy:VisionFlowMLP,
            denoising_steps,
            learn_explore_noise_from,
            inital_noise_scheduler_type,
            min_logprob_denoising_std,
            max_logprob_denoising_std,
            learn_explore_time_embedding,
            time_dim_explore,
            use_time_independent_noise,
            device,
            noise_hidden_dims=None,
            activation_type='Tanh'
    ):
        """
        带噪声的视觉增强FlowMLP模型：在VisionFlowMLP基础上增加可学习的探索噪声
        
        参数说明:
        - policy: VisionFlowMLP, 基础的视觉流匹配策略网络
        - denoising_steps: int, 去噪步数
        - learn_explore_noise_from: int, 从哪一步开始学习探索噪声
        - inital_noise_scheduler_type: str, 初始噪声调度器类型
        - min_logprob_denoising_std: float, 最小对数概率去噪标准差
        - max_logprob_denoising_std: float, 最大对数概率去噪标准差
        - learn_explore_time_embedding: bool, 是否学习探索时间嵌入
        - time_dim_explore: int, 探索时间嵌入维度
        - use_time_independent_noise: bool, 是否使用时间无关噪声
        - device: 设备信息
        - noise_hidden_dims: list, 噪声网络隐藏层维度
        - activation_type: str, 激活函数类型
        """
        super().__init__(
            policy,
            denoising_steps,
            learn_explore_noise_from,
            inital_noise_scheduler_type,
            min_logprob_denoising_std,
            max_logprob_denoising_std,
            learn_explore_time_embedding,
            time_dim_explore,
            use_time_independent_noise,
            device,
            noise_hidden_dims,
            activation_type
        )
    
    def forward(
        self,
        action,
        time,
        cond,
        learn_exploration_noise=False,
        step=-1,
        verbose=False,
        **kwargs,
    )->Tuple[Tensor, Tensor]:
        """
        inputs:
            x: (B, Ta, Da)
            time: (B,) floating point in [0,1) flow matching time
            cond: dict with key state/rgb; more recent obs at the end
                state: (B, To, Do)
            step: (B,) torch.tensor, optional, flow matching inference step, from 0 to denoising_steps-1
            *here, B is the n_envs
        outputs:
             vel                [B, Ta, Da]
             noise_std          [B, Ta x Da]
        """
        B = action.shape[0]
        
        self.policy: VisionFlowMLP
        vel, time_emb, cond_emb = self.policy.forward(action, time, cond, output_embedding=True)
        
        # noise head (for exploration). allow gradient flow.
        if self.initial_noise_scheduler_type=='const' or step < self.learn_explore_noise_from:
            noise_std       = self.logprob_noise_levels[:, step].repeat(B,1)
        else:
            if self.use_time_independent_noise:
                noise_feature    = cond_emb
            else:
                if self.learn_explore_time_embedding:
                    step_ts = torch.tensor(step, device = self.device).repeat(B)
                    time_emb_explore = self.time_embedding_explore(step_ts)
                    noise_feature    = torch.cat([time_emb_explore, cond_emb], dim=-1)
                else:
                    noise_feature    = torch.cat([time_emb.detach(), cond_emb], dim=-1)
            # predict noise
            noise_std = self.explore_noise_net.forward(noise_feature=noise_feature)
        
        return vel, noise_std if learn_exploration_noise else noise_std.detach()
    
