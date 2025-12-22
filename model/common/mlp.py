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
Implementation of Multi-layer Perceptron (MLP).

Residual model is taken from https://github.com/ALRhub/d3il/blob/main/agents/models/common/mlp.py
"""

import torch
from torch import nn
from collections import OrderedDict
import logging


activation_dict = nn.ModuleDict(
    {
        "ReLU": nn.ReLU(),
        "ELU": nn.ELU(),
        "GELU": nn.GELU(),
        "Tanh": nn.Tanh(),
        "Mish": nn.Mish(),
        "Identity": nn.Identity(),
        "Softplus": nn.Softplus(),
        "SiLU": nn.SiLU(),
    }
)


class MLP(nn.Module):
    def __init__(
        self,
        dim_list,
        append_dim=0,
        append_layers=None,
        activation_type="Tanh",
        out_activation_type="Identity",
        use_layernorm=False,
        use_layernorm_final=False,
        dropout=0,
        use_drop_final=False,
        out_bias_init=None,
        verbose=False,
    ):
        super(MLP, self).__init__()

        # Ensure append_layers is always a list to avoid TypeError
        self.append_layers = append_layers if append_layers is not None else []

        # Construct module list
        self.moduleList = nn.ModuleList()
        num_layer = len(dim_list) - 1
        for idx in range(num_layer):
            i_dim = dim_list[idx]
            o_dim = dim_list[idx + 1]
            if append_dim > 0 and idx in self.append_layers:
                i_dim += append_dim
            linear_layer = nn.Linear(i_dim, o_dim)

            # Add module components
            layers = [("linear_1", linear_layer)]
            if use_layernorm and (idx < num_layer - 1 or use_layernorm_final):
                layers.append(("norm_1", nn.LayerNorm(o_dim)))
            if dropout > 0 and (idx < num_layer - 1 or use_drop_final):
                layers.append(("dropout_1", nn.Dropout(dropout)))

            # Add activation function
            act = (
                activation_dict[activation_type]
                if idx != num_layer - 1
                else activation_dict[out_activation_type]
            )
            layers.append(("act_1", act))

            # Re-construct module
            module = nn.Sequential(OrderedDict(layers))
            self.moduleList.append(module)
        if verbose:
            logging.info(self.moduleList)

        # Initialize the bias of the final linear layer if specified
        if out_bias_init is not None:
            final_linear = self.moduleList[-1][0]  # Linear layer is first in the last Sequential
            nn.init.constant_(final_linear.bias, out_bias_init)

    def forward(self, x, append=None):
        for layer_ind, m in enumerate(self.moduleList):
            if append is not None and layer_ind in self.append_layers:
                x = torch.cat((x, append), dim=-1)
            x = m(x)
        return x


class ResidualMLP(nn.Module):
    """
    Simple multi-layer perceptron network with residual connections for
    benchmarking the performance of different networks. The residual layers
    are based on the IBC paper implementation, which uses 2 residual layers
    with pre-activation with or without dropout and normalization.
    
    **残差网络构建说明：**
    1. 输入层：将输入特征映射到隐藏维度
    2. 残差块序列：由多个TwoLayerPreActivationResNetLinear残差块组成
    3. 输出层：将隐藏维度映射到最终输出维度
    4. 可选的最终层归一化和输出激活函数
    
    **网络结构示例：**
    - 输入：dim_list=[64, 256, 256, 256, 10] (输入64维，3个隐藏层，输出10维)
    - 网络组件：输入层(64→256) → 残差块1(256→256) → 输出层(256→10) → 激活
    - 残差块数量：(3-1)/2=1个 (因为隐藏层数量必须为偶数)
    """

    def __init__(
        self,
        dim_list,
        activation_type="Mish",
        out_activation_type="Identity",
        use_layernorm=False,
        use_layernorm_final=False,
        dropout=0,
        out_bias_init=None,
    ):
        super(ResidualMLP, self).__init__()
        hidden_dim = dim_list[1]  # 隐藏层维度
        num_hidden_layers = len(dim_list) - 3  # 隐藏层数量（不包括输入和输出层）
        assert num_hidden_layers % 2 == 0  # 隐藏层数量必须为偶数，因为每个残差块包含两层
        
        # 创建网络层列表，从输入层开始
        self.layers = nn.ModuleList([nn.Linear(dim_list[0], hidden_dim)])
        
        # 添加多个残差块：每个残差块由两个线性层组成
        self.layers.extend(
            [
                TwoLayerPreActivationResNetLinear(
                    hidden_dim=hidden_dim,
                    activation_type=activation_type,
                    use_layernorm=use_layernorm,
                    dropout=dropout,
                )
                for _ in range(1, num_hidden_layers, 2)  # 每两个隐藏层对应一个残差块
            ]
        )
        
        # 添加输出层，将隐藏维度映射到输出维度
        self.layers.append(nn.Linear(hidden_dim, dim_list[-1]))
        
        # 添加可选的最终层归一化
        if use_layernorm_final:
            self.layers.append(nn.LayerNorm(dim_list[-1]))
        
        # 添加输出激活函数
        self.layers.append(activation_dict[out_activation_type])

        # Initialize the bias of the final linear layer if specified
        if out_bias_init is not None:
            for layer in reversed(self.layers):
                if isinstance(layer, nn.Linear):
                    nn.init.constant_(layer.bias, out_bias_init)
                    break

    def forward(self, x):
        """
        前向传播函数
        1. 依次通过所有网络层：输入层 → 残差块 → 输出层 → 激活函数
        2. 残差块内部实现了残差连接，无需在此额外处理
        """
        for layer in self.layers:
            x = layer(x)
        return x


class TwoLayerPreActivationResNetLinear(nn.Module):
    """
    **双线性层预激活残差块**
    
    实现了一个包含两个线性层的预激活残差块，是残差网络的基本构建单元。
    
    **残差连接原理：**
    1. 保存输入x_input
    2. 对输入进行预激活和归一化
    3. 通过两个线性层进行特征变换
    4. 将变换后的特征与原始输入相加（残差连接）：x + 变换(x)
    
    **预激活结构：**
    与传统残差块（[卷积 → 激活 → 卷积] + 残差）不同，
    预激活结构为：[归一化 → 激活 → 卷积 → 归一化 → 激活 → 卷积] + 残差
    这种结构可以避免梯度消失问题，提高网络训练稳定性。
    """
    
    def __init__(
        self,
        hidden_dim,
        activation_type="Mish",
        use_layernorm=False,
        dropout=0,
    ):
        super().__init__()
        # 创建两个线性层，保持隐藏维度不变
        self.l1 = nn.Linear(hidden_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        
        # 激活函数
        self.act = activation_dict[activation_type]
        
        # 可选的层归一化
        if use_layernorm:
            self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-06)
            self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-06)
        
        # 暂不支持dropout
        if dropout > 0:
            raise NotImplementedError("Dropout not implemented for residual MLP!")

    def forward(self, x):
        """
        残差块前向传播
        
        计算流程：
        1. 保存原始输入x_input
        2. 第一层：[归一化 → 激活 → 线性变换]
        3. 第二层：[归一化 → 激活 → 线性变换]
        4. 残差连接：将变换后的结果与原始输入相加
        
        示例：
        输入x → norm1(x) → Mish(x) → l1(x) → norm2(x) → Mish(x) → l2(x) → x + 结果
        """
        x_input = x  # 保存原始输入用于残差连接
        
        # 第一层处理（可选归一化 → 激活 → 线性变换）
        if hasattr(self, "norm1"):
            x = self.norm1(x)  # 可选层归一化
        x = self.l1(self.act(x))  # 激活后进行线性变换
        
        # 第二层处理（可选归一化 → 激活 → 线性变换）
        if hasattr(self, "norm2"):
            x = self.norm2(x)  # 可选层归一化
        x = self.l2(self.act(x))  # 激活后进行线性变换
        
        # 残差连接：将变换后的结果与原始输入相加
        return x + x_input