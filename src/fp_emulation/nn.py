import torch
from torch import nn

from fp_emulation.ozaki import ozaki2_int8_matmul


class Linear(nn.Linear):
    """nn.Linear using INT8 tensor core matmul."""

    def __init__(self, in_features, out_features, bias=True, device=None):
        super().__init__(
            in_features, out_features, bias=bias, device=device, dtype=torch.float64
        )

    def forward(self, x):
        x = x.to(torch.float64)
        shape = x.shape
        out = ozaki2_int8_matmul(x.reshape(-1, shape[-1]), self.weight.T)
        out = out.reshape(*shape[:-1], self.out_features)
        if self.bias is not None:
            out = out + self.bias

        return out


def convert(model):
    """Swap all nn.Linear layers for INT8 tensor core equivalents. Copies weights to fp64."""
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and not isinstance(module, Linear):
            new = Linear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                device=module.weight.device,
            )
            new.weight.data.copy_(module.weight.data.to(torch.float64))
            new.weight.requires_grad_(module.weight.requires_grad)
            if module.bias is not None:
                new.bias.data.copy_(module.bias.data.to(torch.float64))
                new.bias.requires_grad_(module.bias.requires_grad)

            setattr(model, name, new)

        else:
            convert(module)

    return model
