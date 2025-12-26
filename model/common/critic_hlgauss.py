from typing import Union
import torch
from model.common.mlp import MLP, ResidualMLP

class CriticObsHLGauss(torch.nn.Module):
    """State-only critic network. V(s) -> logits (for HL-Gauss classification)"""

    def __init__(
        self,
        cond_dim,
        mlp_dims,
        output_dim, # This will be num_bins
        activation_type="Mish",
        use_layernorm=False,
        residual_style=False,
        out_bias_init=None,  # Set to ensure optimistic initial output
        **kwargs,
    ):
        super().__init__()
        # The output dimension is now output_dim (num_bins) instead of 1
        mlp_dims = [cond_dim] + mlp_dims + [output_dim]
        if residual_style:
            model = ResidualMLP
        else:
            model = MLP
        self.Q1 = model(
            mlp_dims,
            activation_type=activation_type,
            out_activation_type="Identity", # Output logits
            use_layernorm=use_layernorm,
            out_bias_init=out_bias_init,
        )

    def forward(self, cond: Union[dict, torch.Tensor]):
        """
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
            or (B, num_feature) from ViT encoder
        """
        if isinstance(cond, dict):
            B = len(cond["state"])

            # flatten history
            state = cond["state"].view(B, -1)
        else:
            state = cond
        q1 = self.Q1(state)
        # Return logits of shape (B, num_bins)
        return q1
