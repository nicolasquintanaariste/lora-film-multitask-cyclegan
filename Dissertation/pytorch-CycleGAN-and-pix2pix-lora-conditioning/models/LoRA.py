import math
import torch
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


class MultiTaskLoRA(nn.Module):
    """
    Wraps a Conv2d with independent per-task LoRA adapters.

    For each task *i*:  delta_i(x) = lora_B_i(lora_A_i(x))
    Full forward:       module(x)  +  delta_tid(x)

    The shared base module (backbone) is updated by all tasks at every
    iteration.  Each task additionally trains its own low-rank delta so it
    can specialise without interfering with the backbone gradient.

    Initialisation:
        lora_A  — Kaiming-uniform (provides a random projection)
        lora_B  — zeros           (delta is 0 at init → identical to backbone)
    """

    def __init__(self, module: nn.Conv2d, rank: int, num_tasks: int):
        super().__init__()
        if not isinstance(module, nn.Conv2d):
            raise TypeError(
                f"MultiTaskLoRA only wraps nn.Conv2d, got {type(module).__name__}"
            )
        self.module    = module
        self.rank      = rank
        self.num_tasks = num_tasks

        if rank > 0 and num_tasks > 0:
            self.lora_As = nn.ModuleList()
            self.lora_Bs = nn.ModuleList()
            for _ in range(num_tasks):
                lora_A = nn.Conv2d(
                    module.in_channels, rank,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    groups=module.groups,
                    bias=False,
                )
                lora_B = nn.Conv2d(rank, module.out_channels, 1, bias=False)
                nn.init.kaiming_uniform_(lora_A.weight, a=math.sqrt(5))
                nn.init.zeros_(lora_B.weight)
                self.lora_As.append(lora_A)
                self.lora_Bs.append(lora_B)
        else:
            self.lora_As = self.lora_Bs = None

    def forward(self, x, tid=None):
        out = self.module(x)
        if self.lora_As is not None and tid is not None:
            if isinstance(tid, torch.Tensor):
                t = int(tid.item() if tid.numel() == 1 else tid[0].item())
            else:
                t = int(tid)
            out = out + self.lora_Bs[t](self.lora_As[t](x))
        return out