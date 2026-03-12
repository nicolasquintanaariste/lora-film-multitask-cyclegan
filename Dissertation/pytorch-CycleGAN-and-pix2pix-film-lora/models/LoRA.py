import torch.nn as nn

class LoRA(nn.Module):
    def __init__(self, module: nn.Module, rank: int):
        super().__init__()
        self.module = module
        self.is_conv2d = isinstance(module, nn.Conv2d)
        self.is_linear = isinstance(module, nn.Linear)
        if not (self.is_conv2d or self.is_linear):
            raise TypeError(f"LoRA only supports nn.Conv2d or nn.Linear, got {type(module).__name__}")

        if rank > 0:
            # A and B are the low-rank update matrices.
            if self.is_conv2d:
                # Match the wrapped conv spatial transform so LoRA and base outputs align.
                self.lora_A = nn.Conv2d(
                    module.in_channels,
                    rank,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    groups=module.groups,
                    bias=False,
                )
                self.lora_B = nn.Conv2d(rank, module.out_channels, 1, bias=False)
            else:
                self.lora_A = nn.Linear(module.in_features, rank, bias=False)
                self.lora_B = nn.Linear(rank, module.out_features, bias=False)

            # initialise to zero so base behaviour is unchanged
            nn.init.zeros_(self.lora_A.weight)
            nn.init.zeros_(self.lora_B.weight)
        else:
            self.lora_A = self.lora_B = None

    def forward(self, x, *args, **kwargs):
        out = self.module(x, *args, **kwargs)
        if self.lora_A is not None:
            # LoRA update is computed from the same input as the base module.
            out = out + self.lora_B(self.lora_A(x))
        return out