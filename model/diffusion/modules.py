# MIT License
#
# Copyright (c) 2024 Intelligent Robot Motion Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
From Diffuser https://github.com/jannerm/diffuser

For MLP and UNet diffusion models.

"""

import math
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class SinusoidalPosEmb(nn.Module):
    """
    正弦位置编码器 (Sinusoidal Positional Embedding)
    
    该模块将时间步或其他标量值转换为高维特征向量，使用正弦和余弦函数生成位置编码。
    这种编码方式源自Transformer模型，在扩散模型中常用于时间步的嵌入表示。
    
    编码公式：
    PE_(pos,2i) = sin(pos / 10000^(2i/dim))
    PE_(pos,2i+1) = cos(pos / 10000^(2i/dim))
    
    其中 pos 是输入的时间步，dim 是嵌入维度，i 是维度索引。
    
    数值示例：
    假设我们有一个4维的位置编码（dim=4），输入时间步为3（pos=3）：
    1. half_dim = dim // 2 = 2
    2. emb = log(10000) / (half_dim - 1) = log(10000) / 1 ≈ 9.2103
    3. frequencies = exp([0, 1] * -9.2103) = exp([0, -9.2103]) = [1.0000, 0.0001]
    4. pos * frequencies = 3 * [1.0000, 0.0001] = [3.0000, 0.0003]
    5. sin_part = sin([3.0000, 0.0003]) = [0.1411, 0.0003]
    6. cos_part = cos([3.0000, 0.0003]) = [-0.9900, 1.0000]
    7. emb = concat(sin_part, cos_part) = [0.1411, 0.0003, -0.9900, 1.0000]
    
    这样就将标量时间步3转换为了4维的特征向量[0.1411, 0.0003, -0.9900, 1.0000]
    """

    def __init__(self, dim):
        """
        初始化正弦位置编码器
        
        参数:
            dim (int): 输出嵌入向量的维度，必须是偶数
        """
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """
        前向传播函数，将输入标量转换为正弦位置编码
        
        参数:
            x (Tensor): 输入张量，形状为 (batch_size,) 或 (batch_size, 1)
                       通常是时间步信息
            
        返回:
            emb (Tensor): 输出的位置编码，形状为 (batch_size, dim)
                         包含正弦和余弦编码的拼接结果
        
        数值示例：
        假设 dim=4, x=[3, 5] (batch_size=2)
        1. half_dim = 4 // 2 = 2
        2. emb = log(10000) / (2 - 1) = 9.2103
        3. frequencies = exp([0, 1] * -9.2103) = [1.0000, 0.0001]
        4. 对于第一个样本(x=3):
           - pos * frequencies = 3 * [1.0000, 0.0001] = [3.0000, 0.0003]
           - sin_part = sin([3.0000, 0.0003]) = [0.1411, 0.0003]
           - cos_part = cos([3.0000, 0.0003]) = [-0.9900, 1.0000]
        5. 对于第二个样本(x=5):
           - pos * frequencies = 5 * [1.0000, 0.0001] = [5.0000, 0.0005]
           - sin_part = sin([5.0000, 0.0005]) = [-0.9589, 0.0005]
           - cos_part = cos([5.0000, 0.0005]) = [0.2837, 1.0000]
        6. 最终输出:
           - 第一个样本: [0.1411, 0.0003, -0.9900, 1.0000]
           - 第二个样本: [-0.9589, 0.0005, 0.2837, 1.0000]
           - 整体输出形状: (2, 4)
        """
        device = x.device
        # 计算一半的维度数，因为我们将分别生成正弦和余弦编码然后拼接
        half_dim = self.dim // 2
        
        # 计算频率参数的分母：log(10000) / (half_dim - 1)
        # 这是为了创建不同频率的正弦/余弦波
        emb = math.log(10000) / (half_dim - 1)
        
        # 生成指数衰减的频率参数：exp(-i * emb) for i = 0, 1, ..., half_dim-1
        # 这些频率从高频到低频排列
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        
        # 将输入x扩展维度并与频率参数相乘
        # x[:, None]: 形状从 (batch_size,) 变为 (batch_size, 1)
        # emb[None, :]: 形状从 (half_dim,) 变为 (1, half_dim)
        # 结果 emb: 形状为 (batch_size, half_dim)
        emb = x[:, None] * emb[None, :]
        
        # 拼接正弦和余弦编码形成完整的位置编码
        # 最终输出形状: (batch_size, dim) 其中 dim = 2 * half_dim
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """
    Conv1d --> GroupNorm --> Mish
    """

    def __init__(
        self,
        inp_channels,
        out_channels,
        kernel_size,
        n_groups=None,
        activation_type="Mish",
        eps=1e-5,
    ):
        super().__init__()
        if activation_type == "Mish":
            act = nn.Mish()
        elif activation_type == "ReLU":
            act = nn.ReLU()
        else:
            raise "Unknown activation type for Conv1dBlock"

        self.block = nn.Sequential(
            nn.Conv1d(
                inp_channels, out_channels, kernel_size, padding=kernel_size // 2
            ),
            (
                Rearrange("batch channels horizon -> batch channels 1 horizon")
                if n_groups is not None
                else nn.Identity()
            ),
            (
                nn.GroupNorm(n_groups, out_channels, eps=eps)
                if n_groups is not None
                else nn.Identity()
            ),
            (
                Rearrange("batch channels 1 horizon -> batch channels horizon")
                if n_groups is not None
                else nn.Identity()
            ),
            act,
        )

    def forward(self, x):
        return self.block(x)
