import torch.nn as nn

class LoRA(nn.Module):
    def __init__(self, module: nn.Module, rank: int):
        super().__init__()
        self.module = module
        if rank > 0:
            # A and B are the low‑rank update matrices
            self.lora_A = nn.Conv2d(module.in_channels, rank, 1, bias=False)
            self.lora_B = nn.Conv2d(rank, module.out_channels, 1, bias=False)
            # initialise to zero so base behaviour is unchanged
            nn.init.zeros_(self.lora_A.weight)
            nn.init.zeros_(self.lora_B.weight)
        else:
            self.lora_A = self.lora_B = None

    def forward(self, x, *args, **kwargs):
        out = self.module(x, *args, **kwargs)
        if self.lora_A is not None:
            out = out + self.lora_B(self.lora_A(x))
        return out