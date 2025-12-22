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
1-Rectified Flow Policy
"""


import logging
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from collections import namedtuple
from model.flow.mlp_flow import FlowMLP

log = logging.getLogger(__name__)
Sample = namedtuple("Sample", "trajectories chains")

class ReFlow(nn.Module):
    def __init__(
        self,
        network: FlowMLP,
        device: torch.device,
        horizon_steps: int,
        action_dim: int,
        act_min: float,
        act_max: float,
        obs_dim: int,
        max_denoising_steps: int,
        seed: int,
        sample_t_type: str = 'uniform'
    ):
        """Initialize the ReFlow model with specified parameters.

        Args:
            network: FlowMLP network for velocity prediction.
            device: Device to run the model on (e.g., 'cuda' or 'cpu').
            horizon_steps: Number of steps in the trajectory horizon.
            action_dim: Dimension of the action space.
            act_min: Minimum action value for clipping.
            act_max: Maximum action value for clipping.
            obs_dim: Dimension of the observation space.
            max_denoising_steps: Maximum number of denoising steps for sampling.
            seed: Random seed for reproducibility.
            batch_size: Batch size for training and sampling.
            sample_t_type: Type of time sampling ('uniform', 'logitnormal', 'beta').

        Raises:
            ValueError: If max_denoising_steps is not a positive integer.
        """
        super().__init__()
        if int(max_denoising_steps) <= 0:
            raise ValueError('max_denoising_steps must be a positive integer')
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.network = network.to(device)
        self.device = device
        self.horizon_steps = horizon_steps
        self.action_dim = action_dim
        self.data_shape = (self.horizon_steps, self.action_dim)
        self.act_range = (act_min, act_max)
        self.obs_dim = obs_dim
        self.max_denoising_steps = int(max_denoising_steps)
        self.sample_t_type = sample_t_type

    def generate_trajectory(self, x1: Tensor, x0: Tensor, t: Tensor) -> Tensor:
        """Generate rectified flow trajectory xt = t * x1 + (1 - t) * x0.
        生成 Rectified Flow 的轨迹插值 xt = t * x1 + (1 - t) * x0。

        Args:
            x1: Target data tensor of shape (batch_size, horizon_steps, action_dim).
                目标数据张量（真实数据），形状为 (batch_size, horizon_steps, action_dim)。
            x0: Initial noise tensor of shape (batch_size, horizon_steps, action_dim).
                初始噪声张量（高斯噪声），形状为 (batch_size, horizon_steps, action_dim)。
            t: Time step tensor of shape (batch_size,).
                时间步张量，形状为 (batch_size,)，每个样本对应一个时间点 t ∈ [0, 1]。

        Returns:
            Tensor: Interpolated trajectory xt of shape (batch_size, horizon_steps, action_dim).
                插值后的轨迹点 xt，形状同输入数据。
        """
        # 将时间步 t 扩展为与数据 x1 相同的形状，以便进行逐元素运算
        # (batch_size,) -> (batch_size, 1, 1) -> (batch_size, horizon_steps, action_dim)
        t_ = (torch.ones_like(x1, device=self.device) * t.view(x1.shape[0], 1, 1)).to(self.device)  # ReinFlow Authors revised on 04/23/2025
        # 线性插值公式：t=0 时为噪声 x0，t=1 时为数据 x1
        xt = t_ * x1 + (1 - t_) * x0
        return xt

    def sample_time(self, batch_size: int, time_sample_type: str = 'uniform', **kwargs) -> Tensor:
        """Sample time steps from a specified distribution in [0, 1).
        从指定分布中采样时间步 t ∈ [0, 1)。

        Args:
            batch_size: Number of time samples to generate.
                需要生成的样本数量（通常等于 batch_size）。
            time_sample_type: Type of distribution ('uniform', 'logitnormal', 'beta').
                采样分布类型：'uniform' (均匀分布), 'logitnormal' (对数正态), 'beta' (Beta分布)。
            **kwargs: Additional parameters for non-uniform distributions.
                其他参数，用于控制非均匀分布的形状。

        Returns:
            Tensor: Time samples of shape (batch_size,).
                采样得到的时间步张量。

        Raises:
            ValueError: If time_sample_type is not supported.
        """
        supported_time_sample_type = ['uniform', 'logitnormal', 'beta']
        if time_sample_type == 'uniform':
            # 均匀分布采样：t ~ U[0, 1)
            return torch.rand(batch_size, device=self.device)
        elif time_sample_type == 'logitnormal':
            # Logit-Normal 分布采样：更偏向于 0 和 1 两端，有助于训练中间过程
            m = kwargs.get("m", 0)  # Default mean
            s = kwargs.get("s", 1)  # Default standard deviation
            normal_samples = torch.normal(mean=m, std=s, size=(batch_size,), device=self.device)
            logit_normal_samples = (1 / (1 + torch.exp(-normal_samples))).to(self.device)
            return logit_normal_samples
        elif time_sample_type == 'beta':
            # Beta 分布采样
            alpha = kwargs.get("alpha", 1.5)  # Default alpha
            beta = kwargs.get("beta", 1.0)   # Default beta
            s = kwargs.get("s", 0.999)       # Default cutoff
            beta_distribution = torch.distributions.Beta(alpha, beta)
            beta_sample = beta_distribution.sample((batch_size,)).to(self.device)
            tau = s * (1 - beta_sample)
            return tau
        else:
            raise ValueError(f'Unknown time_sample_type = {time_sample_type}. Supported types: {supported_time_sample_type}')

    def generate_target(self, x1: Tensor) -> tuple:
        """Generate training targets for the velocity field.
        生成速度场的训练目标。

        Args:
            x1: Real data tensor of shape (batch_size, horizon_steps, action_dim).
                真实数据样本。

        Returns:
            tuple: Contains (xt, t, obs) and v where:
                - xt: Corrupted data tensor of shape (batch_size, horizon_steps, action_dim).
                    被加噪后的中间状态数据（作为网络输入）。
                - t: Time step tensor of shape (batch_size,).
                    采样的时间步（作为网络输入）。
                - v: Target velocity tensor of shape (batch_size, horizon_steps, action_dim).
                    目标速度场（网络需要预测的标签），计算公式 v = x1 - x0。
        """
        # 1. 采样时间步 t
        t = self.sample_time(batch_size=x1.shape[0], time_sample_type=self.sample_t_type)
        # 2. 生成初始噪声 x0 (标准正态分布)
        x0 = torch.randn(x1.shape, dtype=torch.float32, device=self.device)
        # 3. 计算中间状态 xt = t*x1 + (1-t)*x0
        xt = self.generate_trajectory(x1, x0, t)
        # 4. 计算目标速度 v = d(xt)/dt = x1 - x0
        # 因为 xt = t*x1 + (1-t)*x0 = x0 + t*(x1-x0)，对 t 求导即得 x1 - x0
        v = x1 - x0
        return (xt, t), v

    def loss(self, xt: Tensor, t: Tensor, obs: dict, v: Tensor) -> Tensor:
        """Compute the MSE loss between predicted and target velocities.
        计算预测速度与目标速度之间的均方误差 (MSE) 损失。

        Args:
            xt: Corrupted data tensor of shape (batch_size, horizon_steps, action_dim).
                当前时间步的噪声数据。
            t: Time step tensor of shape (batch_size,).
                当前时间步。
            obs: Dictionary containing 'state' tensor of shape (batch_size, cond_steps, obs_dim).
                观测条件（Condition），例如历史状态。
            v: Target velocity tensor of shape (batch_size, horizon_steps, action_dim).
                真实的目标速度向量 (x1 - x0)。

        Returns:
            Tensor: Mean squared error loss.
                MSE 损失值。
        """
        # 网络预测速度场 v_hat = Network(xt, t, condition)
        v_hat = self.network(xt, t, obs)
        # 计算 MSE 损失：||v_hat - v||^2
        return F.mse_loss(input=v_hat, target=v)
    
    @torch.no_grad()
    def sample(
        self,
        cond: dict,
        inference_steps: int,
        record_intermediate: bool = False,
        clip_intermediate_actions: bool = True,
        z:torch.Tensor = None
    ) -> Sample:
        """Sample trajectories using the learned velocity field.
        使用学习到的速度场进行采样（推理），生成轨迹。通常使用欧拉法（Euler Method）求解 ODE。

        Args:
            cond: Dictionary containing 'state' tensor of shape (batch_size, cond_steps, obs_dim).
                条件输入（观测状态）。
            inference_steps: Number of denoising steps.
                推理步数（去噪步数），步数越多精度越高但速度越慢。
            record_intermediate: Whether to return intermediate predictions.
                是否记录中间过程的轨迹。
            clip_intermediate_actions: Whether to clip actions to act_range.
                是否将生成的动作截断到有效范围内。

        Returns:
            Sample: Named tuple with 'trajectories' (and 'chains' if record_intermediate).
                包含生成的轨迹结果。
        """
        B = cond['state'].shape[0]
        if record_intermediate:
            x_hat_list = torch.zeros((inference_steps,) + self.data_shape, device=self.device)
        
        # 1. 初始化状态：从标准正态分布采样 x0，或者使用指定的噪声 z (B, horizon_steps, action_dim)
        x_hat = z if z is not None else torch.randn((B,) + self.data_shape, device=self.device)
        
        # 2. 定义时间步长 dt = 1 / N
        dt = (1 / inference_steps) * torch.ones_like(x_hat, device=self.device)
        
        # 3. 生成时间序列 steps = [0, 1/N, 2/N, ..., (N-1)/N]
        steps = torch.linspace(0, 1-1/inference_steps, inference_steps, device=self.device).repeat(B, 1)
        
        # 4. 逐步求解 ODE: dx/dt = v(x, t) -> x_{t+dt} = x_t + v(x_t, t) * dt
        for i in range(inference_steps):
            t = steps[:, i]
            # 预测当前时刻的速度场
            vt = self.network(x_hat, t, cond)
            # 欧拉更新：当前位置 + 速度 * 时间步长
            x_hat += vt * dt
            
            # 截断动作值，防止超出物理限制
            if clip_intermediate_actions or i == inference_steps-1: # always clip the output action. appended by ReinFlow Authors on 04/25/2025
                x_hat = x_hat.clamp(*self.act_range)
            
            if record_intermediate:
                x_hat_list[i] = x_hat
                
        return Sample(trajectories=x_hat, chains=x_hat_list if record_intermediate else None)