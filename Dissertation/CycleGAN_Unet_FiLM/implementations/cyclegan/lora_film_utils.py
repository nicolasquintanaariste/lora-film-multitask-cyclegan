import torch
import torch.nn as nn

from FiLM import FiLM


@torch.no_grad()
def init_generic_film_embeddings(model: nn.Module, generic_id: int):
    """
    Initialise the new FiLM embedding row (generic_id) as the mean
    of the existing base-task rows.

    Assumes model FiLM modules use:
        self.embed = nn.Embedding(num_tasks, emb_dim)
    """
    for m in model.modules():
        if isinstance(m, FiLM):
            w = m.embed.weight  # [num_tasks, emb_dim]

            # Only initialise if this module has the extra row
            if w.size(0) == generic_id + 1:
                base_rows = w[:generic_id]          # existing task rows
                w[generic_id].copy_(base_rows.mean(dim=0))


def freeze_all_except_lora_and_film(model: nn.Module, train_film_mlp: bool = True):
    """
    Freeze all parameters except:
        - LoRA parameters (lora_A / lora_B)
        - FiLM embedding weights
        - optionally FiLM MLP (to_gamma_beta)

    Args:
        model: Generator model
        train_film_mlp: whether to also train FiLM MLP layers
    """

    # Freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze LoRA parameters
    for name, p in model.named_parameters():
        lname = name.lower()
        if "lora_a" in lname or "lora_b" in lname:
            p.requires_grad = True

    # Unfreeze FiLM parameters
    for m in model.modules():
        if isinstance(m, FiLM):
            # Always train embedding (so generic row can adapt)
            m.embed.weight.requires_grad = True

            # Optionally train FiLM MLP
            if train_film_mlp:
                for p in m.to_gamma_beta.parameters():
                    p.requires_grad = True
                    
def load_state_skip_film_embeddings(model, ckpt_path, map_location="cpu"):
    sd = torch.load(ckpt_path, map_location=map_location)

    # Remove all FiLM embedding weights (they are task-count dependent)
    keys_to_drop = [k for k in sd.keys() if k.endswith("film.embed.weight") or k.endswith("bottleneck_film.embed.weight")]
    for k in keys_to_drop:
        sd.pop(k, None)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load_state_skip_film_embeddings] dropped={len(keys_to_drop)} missing={len(missing)} unexpected={len(unexpected)}")
    return missing, unexpected