# save_utils.py

import os
import datetime
import torch
import json

def save_final_models(local_model_folder, G_AB, G_BA, D_A, D_B, opt):
    final_dir = os.path.join(local_model_folder, "final_model")
    os.makedirs(final_dir, exist_ok=True)

    torch.save(G_AB.state_dict(), os.path.join(final_dir, f"G_AB_{opt.n_epochs}.pth"))
    torch.save(G_BA.state_dict(), os.path.join(final_dir, f"G_BA_{opt.n_epochs}.pth"))
    torch.save(D_A.state_dict(), os.path.join(final_dir, f"D_A_{opt.n_epochs}.pth"))
    torch.save(D_B.state_dict(), os.path.join(final_dir, f"D_B_{opt.n_epochs}.pth"))

    if opt.lora:
        lora_state_dict = {name: param.detach().cpu()
                        for name, param in G_AB.named_parameters()
                        if param.requires_grad}
        torch.save(lora_state_dict, os.path.join(local_model_folder, "G_AB_lora.pth"))
        
        print("LoRA keys sample:", [k for k in G_AB.state_dict().keys() if "lora" in k.lower()][:10])
        
def save_run_summary(save_path: str, summary: dict):
    details_dir = os.path.join(save_path, "details")
    os.makedirs(details_dir, exist_ok=True)    

    with open(os.path.join(details_dir, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
       
def save_hyperparameters(opt, save_path):
    """Save hyperparameters to JSON file."""
    details_dir = os.path.join(save_path, "details")
    os.makedirs(details_dir, exist_ok=True)

    out_path = os.path.join(details_dir, "hyperparams.json")
    with open(out_path, "w") as f:
        json.dump(vars(opt), f, indent=4)
        
def save_model_checkpoints(checkpoint_folder, epoch, G_AB, G_BA, D_A, D_B):
        # Remove previous checkpoints
        for f in os.listdir(checkpoint_folder):
            file_path = os.path.join(checkpoint_folder, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
                
        # Save checkpoints
        torch.save(G_AB.state_dict(), os.path.join(checkpoint_folder, f"G_AB_{epoch}.pth"))
        torch.save(G_BA.state_dict(), os.path.join(checkpoint_folder, f"G_BA_{epoch}.pth"))
        torch.save(D_A.state_dict(), os.path.join(checkpoint_folder, f"D_A_{epoch}.pth"))
        torch.save(D_B.state_dict(), os.path.join(checkpoint_folder, f"D_B_{epoch}.pth"))
        
        print(f" Models saved → {checkpoint_folder}", flush=True)
