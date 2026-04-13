import torch
import torch.nn as nn

class FiLM(nn.Module):
    def __init__(self, num_tasks: int, num_channels: int, emb_dim: int = 64):
        super().__init__()
        self.embed = nn.Embedding(num_tasks, emb_dim)
        self.to_gamma_beta = nn.Sequential(
            nn.Linear(emb_dim, 2 * num_channels),
            nn.SiLU(),
            nn.Linear(2 * num_channels, 2 * num_channels),
        )

        # Optional: start close to identity modulation
        nn.init.zeros_(self.to_gamma_beta[-1].weight)
        nn.init.zeros_(self.to_gamma_beta[-1].bias)

    def forward(self, x, task_id):
        """
        x: [B, C, H, W]
        task_id: [B] or [1] long, or int
        """
        if not isinstance(task_id, torch.Tensor):
            task_id = torch.tensor(task_id, dtype=torch.long, device=x.device)
        if task_id.dim() == 0:
            task_id = task_id.unsqueeze(0)
        if task_id.size(0) == 1 and x.size(0) > 1:
            task_id = task_id.expand(x.size(0))

        e = self.embed(task_id)                       # [B, emb_dim]
        gb = self.to_gamma_beta(e)                    # [B, 2C]
        gamma, beta = gb.chunk(2, dim=1)              # [B, C], [B, C]
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)     # [B, C, 1, 1]
        beta  = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + gamma) + beta               # (1+gamma) helps stability
