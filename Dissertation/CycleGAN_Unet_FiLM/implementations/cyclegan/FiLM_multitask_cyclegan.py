import argparse
import os
import shutil
import numpy as np
import math
import itertools
import datetime
import time
import random
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *
from utils import *

import torch.nn as nn
import torch.nn.functional as F
import torch
import gc

from loss_utils import LossLogger, plot_losses
from metrics_utils import MetricLogger, plot_fid, plot_kid, compute_fid_kid, fid_save_real

from IPython.display import clear_output

import warnings

warnings.filterwarnings(
    "ignore",
    message="TypedStorage is deprecated",
    category=UserWarning,
)

def main():

    # Base folder
    base_folder = os.path.join("Dissertation", "CycleGAN_Unet_FiLM")
    data_root = os.path.join("Dissertation", "data")

    start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=4, help="number of epochs of training")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["day2night", "horse2zebra", "summer2winter_yosemite", "monet2photo"],
        help="List of tasks/datasets to train on (space-separated)",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=1, help="epoch from which to start lr decay")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_height", type=int, default=64, help="size of image height")
    parser.add_argument("--img_width", type=int, default=64, help="size of image width")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=500, help="interval between saving generator outputs")
    parser.add_argument("--fid_interval", type=int, default=1, help="interval between fid calculation")
    parser.add_argument("--checkpoint_interval", type=int, default=5, help="interval between saving model checkpoints")
    parser.add_argument("--n_residual_blocks", type=int, default=3, help="number of residual blocks in generator")
    parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
    parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument(
        "--data_folder",
        type=str,
        default=data_root,
        help="folder from which the data is retreived from"
    )
    parser.add_argument(
        "--session_folder",
        type=str,
        default=base_folder,
        help="directory of colab session storage to save model data"
    )
    parser.add_argument("--save_model", action="store_true", help="save model data from colab session storage")
    parser.add_argument(
        "--checkpoint_model",
        type=str,
        default=None,
        help="checkpoint to start training from. i.e saved_checkpoint/day2night"
    )
    parser.add_argument("--lora", action="store_true", help="fine tune using LoRA adapters")
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default=None,
        help="pretrained model to finetune lora from. i.e saved_models/day2night/model_20260128_122440"
    )

    opt = parser.parse_args()

    # Create directories
    task_name = "-".join(opt.tasks)
    suffix = "_lora" if opt.lora else ""

    local_model_folder = os.path.join(base_folder, "saved_models", task_name + suffix, f"model_{start_time}")
    os.makedirs(local_model_folder, exist_ok=True)

    session_model_folder = os.path.join(opt.session_folder, "saved_models", task_name + suffix, f"model_{start_time}")
    os.makedirs(session_model_folder, exist_ok=True)

    image_folder = os.path.join(session_model_folder, "images")
    fid_image_dir = os.path.join(session_model_folder, "images", "fake")
    fid_image_dir_A = os.path.join(fid_image_dir, "A")
    fid_image_dir_B = os.path.join(fid_image_dir, "B")
    
    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(fid_image_dir, exist_ok=True)
    os.makedirs(fid_image_dir, exist_ok=True)
    os.makedirs(fid_image_dir_A, exist_ok=True)
    os.makedirs(fid_image_dir_B, exist_ok=True)

    checkpoint_folder = os.path.join(base_folder, "saved_checkpoints", task_name + suffix)
    os.makedirs(checkpoint_folder, exist_ok=True)

    # Losses
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    cuda = torch.cuda.is_available()
    print("CUDA available:", cuda)
    print("Using data folder:", opt.data_folder)

    input_shape = (opt.channels, opt.img_height, opt.img_width)

    # Image transformations
    transforms_ = [
        transforms.Resize(int(opt.img_height * 1.12), Image.BICUBIC),
        transforms.RandomCrop((opt.img_height, opt.img_width)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    
    # Metric plots
    real_dir = os.path.join(opt.data_folder, opt.tasks[0], "test", "B_normalised")
    
    fid_save_real(
        in_dir=os.path.join(opt.data_folder, opt.tasks[0]),
        out_dir=real_dir,
        transforms_=transforms_,
        max_images=250,
        batch_size=opt.batch_size,
        n_cpu=opt.n_cpu
    )
    
    fidkid_csv = os.path.join(local_model_folder, "fid_kid_log.csv")
    fidkid_plot_path = os.path.join(local_model_folder, "fid_kid_epoch.png")
    fid_task = opt.tasks[0]  # only the first task
    metric_logger = MetricLogger(csv_path=os.path.join(local_model_folder, "fid_kid.csv"))

    # Loss plots
    os.makedirs(session_model_folder, exist_ok=True)

    loss_csv = os.path.join(local_model_folder, "loss_log.csv")
    loss_plot_path = os.path.join(local_model_folder, "loss_plot.png")
    logger = LossLogger(csv_path=loss_csv)

    # Initialize generator and discriminator
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # LoRA report
    def lora_report(model, name="G"):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        lora_modules = sum(1 for _, m in model.named_modules() if "lora" in type(m).__name__.lower())
        lora_trainable = sum(p.numel() for n, p in model.named_parameters()
                            if p.requires_grad and "lora" in n.lower())
        print(f"[LoRA:{name}] mods={lora_modules} train={trainable:,}/{total:,} ({trainable/total*100:.3f}%) "
            f"lora_train={lora_trainable:,}")

    def lora_grad_mean(model):
        grads = [p.grad.detach().abs().mean().item()
                for n, p in model.named_parameters()
                if "lora" in n.lower() and p.grad is not None]
        return float(np.mean(grads)) if grads else 0.0
    
    # Task ids for FiLM task conditioning
    task2id = {t: i for i, t in enumerate(opt.tasks)}
    num_tasks = len(opt.tasks)

    G_AB = GeneratorUNet(input_shape, num_tasks=num_tasks, film_emb_dim=64)
    G_BA = GeneratorUNet(input_shape, num_tasks=num_tasks, film_emb_dim=64)

    D_A = Discriminator(input_shape, num_tasks=num_tasks, film_emb_dim=64)
    D_B = Discriminator(input_shape, num_tasks=num_tasks, film_emb_dim=64)

    print("Generator parameters (G_AB, G_BA, G_AB + G_BA):", count_parameters(G_AB), count_parameters(G_AB), count_parameters(G_AB) + count_parameters(G_BA))
    print("Discriminator parameters (D_A, D_B, D_A + D_B):", count_parameters(D_A), count_parameters(D_B), count_parameters(D_A) + count_parameters(D_B))

    if cuda:
        G_AB = G_AB.cuda()
        G_BA = G_BA.cuda()
        D_A = D_A.cuda()
        D_B = D_B.cuda()
        criterion_GAN.cuda()
        criterion_cycle.cuda()
        criterion_identity.cuda()

    if opt.checkpoint_model is not None:
        print(f"Resuming training from epoch {opt.epoch}")
        G_AB.load_state_dict(torch.load(os.path.join(opt.checkpoint_model, f"G_AB_{opt.epoch}.pth")))
        G_BA.load_state_dict(torch.load(os.path.join(opt.checkpoint_model, f"G_BA_{opt.epoch}.pth")))
        D_A.load_state_dict(torch.load(os.path.join(opt.checkpoint_model, f"D_A_{opt.epoch}.pth")))
        D_B.load_state_dict(torch.load(os.path.join(opt.checkpoint_model, f"D_B_{opt.epoch}.pth")))
        
    elif opt.lora:
        # Fine tune frozen network with loRA
        pretrained_path = f"{base_folder}/{opt.pretrained_model}"
        G_AB.load_state_dict(torch.load(pretrained_path + "/G_AB_final.pth"))
        G_BA.load_state_dict(torch.load(pretrained_path + "/G_BA_final.pth"))
        
        G_AB = apply_lora_to_unet(G_AB, rank=4, alpha=1.0)
        G_BA = apply_lora_to_unet(G_BA, rank=4, alpha=1.0)
        
        lora_report(G_AB, "G_AB")
        lora_report(G_BA, "G_BA")
    else:
        # Initialize weights
        G_AB.apply(weights_init_normal)
        G_BA.apply(weights_init_normal)
        D_A.apply(weights_init_normal)
        D_B.apply(weights_init_normal)


    # Optimizers
    optimizer_G = torch.optim.Adam(
        itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
    )
    if opt.lora:
        lora_params = [p for p in G_AB.parameters() if p.requires_grad] + \
                    [p for p in G_BA.parameters() if p.requires_grad]
        optimizer_G = torch.optim.Adam(lora_params, lr=opt.lr, betas=(opt.b1, opt.b2))
        
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # Learning rate update schedulers
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
    )
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
    )
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
    )

    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    # Buffers of previously generated samples
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # Training data loader
    task_loaders = {}
    task_iters = {}

    for task in opt.tasks:
        print(f"Retreiving data from: {os.path.join(opt.data_folder, task)}")
        print(os.listdir(os.path.join(opt.data_folder, task)))
        
        dataset = ImageDataset(
            root=os.path.join(opt.data_folder, task),
            transforms_=transforms_,
            unaligned=True,
            mode="train"
        )

        loader = DataLoader(
            dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_cpu,
            drop_last=True
        )

        task_loaders[task] = loader
        task_iters[task] = iter(loader)

    # Test data loader
    val_loaders = {}
    for task in opt.tasks:
        val_dataset = ImageDataset(
            root=os.path.join(opt.data_folder, task),
            transforms_=transforms_,
            unaligned=True,
            mode="test"
        )
        val_loaders[task] = DataLoader(val_dataset, batch_size=5, shuffle=True, num_workers=0)
        
    def infer(loader, tid):
        G_AB.eval()
        G_BA.eval()
        
        try:
            imgs = next(iter(loader))
        except StopIteration:
            # Re-create iterator if exhausted
            loader_iter = iter(loader)
            imgs = next(loader_iter)
        
        real_A = Variable(imgs["A"].type(Tensor))
        real_B = Variable(imgs["B"].type(Tensor))
        with torch.no_grad():
            fake_B = G_AB(real_A, tid)
            fake_A = G_BA(real_B, tid)
            
        return fake_A, fake_B, real_A, real_B
        
    def sample_images(batches_done):
        """Saves a generated sample from the validation loaders for all tasks."""
        G_AB.eval()
        G_BA.eval()
        
        for task, loader in val_loaders.items():
            tid = torch.tensor([task2id[task]], device=next(G_AB.parameters()).device, dtype=torch.long)
            fake_A, fake_B, real_A, real_B = infer(loader, tid)
            
            # Arrange images along x-axis
            real_A_grid = make_grid(real_A, nrow=5, normalize=True)
            fake_B_grid = make_grid(fake_B, nrow=5, normalize=True)
            real_B_grid = make_grid(real_B, nrow=5, normalize=True)
            fake_A_grid = make_grid(fake_A, nrow=5, normalize=True)
            
            # Arrange images along y-axis: [real_A | fake_B | real_B | fake_A]
            image_grid = torch.cat((real_A_grid, fake_B_grid, real_B_grid, fake_A_grid), 1)
            
            # Save image with task name included
            save_image(
                image_grid,
                os.path.join(image_folder, f"{task}_{batches_done}.png"),
                normalize=False
            )
            
    def fid_inference(epoch):
        # FID data loader
        fid_max_imgs = 250
        fid_dataset = ImageDataset(
            root=os.path.join(opt.data_folder, opt.tasks[0]),
            transforms_=transforms_,
            unaligned=True,
            mode="test"
        )
        fid_loader = DataLoader(fid_dataset, batch_size=fid_max_imgs, shuffle=True, num_workers=0)
           
        gc.collect()
        torch.cuda.empty_cache()
        tid = torch.tensor([task2id[opt.tasks[0]]], device=next(G_AB.parameters()).device, dtype=torch.long)
        fake_A, fake_B, real_A, real_B = infer(fid_loader, tid)
        
        epoch_dir_A = os.path.join(fid_image_dir_A, f"epoch{epoch:03d}")
        epoch_dir_B = os.path.join(fid_image_dir_B, f"epoch{epoch:03d}")
        os.makedirs(epoch_dir_A, exist_ok=True)
        os.makedirs(epoch_dir_B, exist_ok=True)

        fake_A = (fake_A.detach().cpu() * 0.5 + 0.5)  # map [-1,1] -> [0,1]
        fake_B = (fake_B.detach().cpu() * 0.5 + 0.5)

        for i in range(fake_A.size(0)):
            save_image(fake_A[i], os.path.join(epoch_dir_A, f"{i:04d}.png"), normalize=False)
            save_image(fake_B[i], os.path.join(epoch_dir_B, f"{i:04d}.png"), normalize=False)      

    # ----------
    #  Training
    # ----------
    def get_batch(task):
        try:
            batch = next(task_iters[task])
        except StopIteration:
            task_iters[task] = iter(task_loaders[task])
            batch = next(task_iters[task])
        return batch

    prev_time = time.time()
    for epoch in range(opt.epoch, opt.n_epochs):
        steps_per_epoch = max(len(loader) for loader in task_loaders.values())

        balanced_tasks = opt.tasks * (steps_per_epoch // len(opt.tasks) + 1)
        balanced_tasks = balanced_tasks[:steps_per_epoch]  # trim to exact steps
        random.shuffle(balanced_tasks)  # shuffle order within epoch

        for step, task in enumerate(balanced_tasks):
            batches_done = epoch * steps_per_epoch + step
            batch = get_batch(task)
            
            # Set model input
            real_A = Variable(batch["A"].type(Tensor))
            real_B = Variable(batch["B"].type(Tensor))
            
            # Task id
            tid = torch.tensor([task2id[task]], device=real_A.device, dtype=torch.long)

            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((real_A.size(0), *D_A.output_shape))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((real_A.size(0), *D_A.output_shape))), requires_grad=False)

            # ------------------
            #  Train Generators
            # ------------------

            G_AB.train()
            G_BA.train()

            optimizer_G.zero_grad()

            # Identity loss
            loss_id_A = criterion_identity(G_BA(real_A, tid), real_A)
            loss_id_B = criterion_identity(G_AB(real_B, tid), real_B)

            loss_identity = (loss_id_A + loss_id_B) / 2

            # GAN loss
            fake_B = G_AB(real_A, tid)
            loss_GAN_AB = criterion_GAN(D_B(fake_B, tid), valid)
            fake_A = G_BA(real_B, tid)
            loss_GAN_BA = criterion_GAN(D_A(fake_A, tid), valid)

            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            # Cycle loss
            recov_A = G_BA(fake_B, tid)
            loss_cycle_A = criterion_cycle(recov_A, real_A)
            recov_B = G_AB(fake_A, tid)
            loss_cycle_B = criterion_cycle(recov_B, real_B)

            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            # Total loss
            loss_G = loss_GAN + opt.lambda_cyc * loss_cycle + opt.lambda_id * loss_identity

            loss_G.backward()
            
            # LoRA learning check
            if opt.lora and (batches_done % 2 == 0):
                print(f" \n [LoRA] grad_mean AB={lora_grad_mean(G_AB):.2e} BA={lora_grad_mean(G_BA):.2e}")
            
            optimizer_G.step()

            # -----------------------
            #  Train Discriminator A
            # -----------------------

            optimizer_D_A.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_A(real_A, tid), valid)
            # Fake loss (on batch of previously generated samples)
            fake_A_ = fake_A_buffer.push_and_pop(fake_A)
            loss_fake = criterion_GAN(D_A(fake_A_.detach(), tid), fake)
            # Total loss
            loss_D_A = (loss_real + loss_fake) / 2

            loss_D_A.backward()
            optimizer_D_A.step()

            # -----------------------
            #  Train Discriminator B
            # -----------------------

            optimizer_D_B.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_B(real_B, tid), valid)
            # Fake loss (on batch of previously generated samples)
            fake_B_ = fake_B_buffer.push_and_pop(fake_B)
            loss_fake = criterion_GAN(D_B(fake_B_.detach(), tid), fake)
            # Total loss
            loss_D_B = (loss_real + loss_fake) / 2

            loss_D_B.backward()
            optimizer_D_B.step()

            loss_D = (loss_D_A + loss_D_B) / 2

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_left = opt.n_epochs * steps_per_epoch - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()
            
            # Graph log
            with torch.no_grad():
                dA_real_mean = D_A(real_A, tid).mean().item()
                dA_fake_mean = D_A(fake_A.detach(), tid).mean().item()
                dB_real_mean = D_B(real_B, tid).mean().item()
                dB_fake_mean = D_B(fake_B.detach(), tid).mean().item()

            logger.log(
                step=batches_done,
                loss_G=loss_G.item(),
                loss_GAN=loss_GAN.item(),
                loss_cycle=loss_cycle.item(),
                loss_identity=loss_identity.item(),
                loss_D=loss_D.item(),
                loss_D_A=loss_D_A.item(),
                loss_D_B=loss_D_B.item(),
                dA_real_mean=dA_real_mean,
                dA_fake_mean=dA_fake_mean,
                dB_real_mean=dB_real_mean,
                dB_fake_mean=dB_fake_mean,
            )           

            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s"
                % (
                    epoch + 1,
                    opt.n_epochs,
                    step + 1,
                    steps_per_epoch,
                    loss_D.item(),
                    loss_G.item(),
                    loss_GAN.item(),
                    loss_cycle.item(),
                    loss_identity.item(),
                    time_left,
                )
            )

            # # If at sample interval save image
            # if batches_done % opt.sample_interval == 0:
            #     sample_images(batches_done)
            #     plot_losses(logger, out_path=loss_plot_path, smooth_alpha=0.1, last_n=None, show=False)      

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()
        
        clear_output(wait=True)
        
        # Plot FID
        if epoch % opt.fid_interval == 0:
            fid_inference(epoch)
            fake_dir = os.path.join(fid_image_dir_B, f"epoch{epoch:03d}")
            fid, kid = compute_fid_kid(real_dir, fake_dir)

            metric_logger.log(epoch=epoch + 1, fid=fid, kid=kid)
            plot_fid(
                metric_logger,
                out_path=os.path.join(local_model_folder, f"fid_{fid_task}.png"),
                show=False
            )
            plot_kid(
                metric_logger,
                out_path=os.path.join(local_model_folder, f"kid_{fid_task}.png"),
                show=False
            )
        
        # Generate samples and plot losses
        if epoch % opt.sample_interval == 0:
            sample_images(epoch)
            plot_losses(logger, out_path=loss_plot_path, smooth_alpha=0.1, last_n=None, show=False)      

        
        # Copy model folder to drive
        if opt.save_model and os.path.abspath(opt.session_folder) != os.path.abspath(base_folder):
            destination = os.path.join(base_folder, "saved_models", task_name + suffix, f"model_{start_time}")
            copy_missing(session_model_folder, destination)

        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # Remove previous checkpoints
            for f in os.listdir(checkpoint_folder):
                file_path = os.path.join(checkpoint_folder, f)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        
            # Save model checkpoints
            torch.save(G_AB.state_dict(), os.path.join(checkpoint_folder, f"G_AB_{epoch}.pth"))
            torch.save(G_BA.state_dict(), os.path.join(checkpoint_folder, f"G_BA_{epoch}.pth"))
            torch.save(D_A.state_dict(), os.path.join(checkpoint_folder, f"D_A_{epoch}.pth"))
            torch.save(D_B.state_dict(), os.path.join(checkpoint_folder, f"D_B_{epoch}.pth"))
            
            print(f" Models saved → {checkpoint_folder}", flush=True)

    # -----------------------------
    # Save final model with timestamp
    # -----------------------------
    torch.save(G_AB.state_dict(), os.path.join(session_model_folder, "G_AB_final.pth"))
    torch.save(G_BA.state_dict(), os.path.join(session_model_folder, "G_BA_final.pth"))
    torch.save(D_A.state_dict(), os.path.join(session_model_folder, "D_A_final.pth"))
    torch.save(D_B.state_dict(), os.path.join(session_model_folder, "D_B_final.pth"))

    if opt.lora:
        lora_state_dict = {name: param.detach().cpu()
                        for name, param in G_AB.named_parameters()
                        if param.requires_grad}
        torch.save(lora_state_dict, os.path.join(session_model_folder, "G_AB_lora.pth"))
        
        print("LoRA keys sample:", [k for k in G_AB.state_dict().keys() if "lora" in k.lower()][:10])

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()