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

from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *
from utils import *
from save_utils import *
from config import parse_args

import torch.nn as nn
import torch.nn.functional as F
import torch

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

    # Parse input parameters
    opt = parse_args(base_folder, data_root)
    
    # Start time
    start_time = datetime.datetime.now()
    timer = PhaseTimer(use_cuda_sync=True)
    
    # Seed everything
    seed_everything(opt.seed)

    # Create directories
    training_tasks = opt.lora if opt.lora is not None else opt.tasks
    task_name = "-".join(training_tasks)
    suffix = "_lora" if opt.lora is not None else ""

    start_time_str = start_time.strftime("%Y%m%d_%H%M%S")
    local_model_folder = os.path.join(base_folder, "saved_models", task_name + suffix, f"model_{start_time_str}")
    os.makedirs(local_model_folder, exist_ok=True)

    session_model_folder = os.path.join(opt.session_folder, "saved_models", task_name + suffix, f"model_{start_time_str}")
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

    checkpoint_folder = os.path.join(session_model_folder, "saved_checkpoints")
    os.makedirs(checkpoint_folder, exist_ok=True)
    
    # Save hyperparameters
    save_hyperparameters(opt, session_model_folder)

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
        # transforms.Resize(int(opt.img_height * 1.12), Image.BICUBIC), # x1.12 would make img bigger and crop edges
        transforms.Resize(opt.img_height),  # Resize shortest side to img_height, maintains aspect ratio
        transforms.RandomCrop((opt.img_height, opt.img_width)),  # Now crop to square
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    
    # Metric plots
    real_dir = os.path.join(opt.data_folder, training_tasks[0], "test", "B_normalised")
    
    with timer.track("prep/fid_save_real"):
        fid_save_real(
            in_dir=os.path.join(opt.data_folder, training_tasks[0]),
            out_dir=real_dir,
            transforms_=transforms_,
            max_images=250,
            batch_size=opt.batch_size,
            n_cpu=opt.n_cpu
        )
    
    fidkid_csv = os.path.join(local_model_folder, "fid_kid_log.csv")
    fidkid_plot_path = os.path.join(local_model_folder, "fid_kid_epoch.png")
    fid_task = training_tasks[0]  # only the first training task
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

    param_summary = {
        "G_AB": int(count_parameters(G_AB)),
        "G_BA": int(count_parameters(G_BA)),
        "D_A":  int(count_parameters(D_A)),
        "D_B":  int(count_parameters(D_B)),
    }

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
        G_AB.load_state_dict(torch.load(pretrained_path + "/final_model/G_AB_final.pth"))
        G_BA.load_state_dict(torch.load(pretrained_path + "/final_model/G_BA_final.pth"))
        
        G_AB = apply_lora_to_unet(G_AB, rank=4, alpha=1.0)
        G_BA = apply_lora_to_unet(G_BA, rank=4, alpha=1.0)
        
        # Move LoRA parameters to CUDA if needed
        if cuda:
            G_AB = G_AB.cuda()
            G_BA = G_BA.cuda()
        
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
        itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=opt.lr_G, betas=(opt.b1, opt.b2)
    )
    if opt.lora:
        lora_params = [p for p in G_AB.parameters() if p.requires_grad] + \
                    [p for p in G_BA.parameters() if p.requires_grad]
        optimizer_G = torch.optim.Adam(lora_params, lr=opt.lr_G, betas=(opt.b1, opt.b2))
        
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr_D, betas=(opt.b1, opt.b2))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr_D, betas=(opt.b1, opt.b2))

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
    fake_A_buffers = {t: ReplayBuffer() for t in training_tasks}
    fake_B_buffers = {t: ReplayBuffer() for t in training_tasks}

    # Training data loader
    task_loaders = {}
    task_iters = {}
    
    # Define generator for seeding
    g = torch.Generator()
    g.manual_seed(opt.seed)

    for task in training_tasks:
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
            drop_last=True,
            worker_init_fn=seed_worker,
            generator=g
        )

        task_loaders[task] = loader
        task_iters[task] = iter(loader)

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
        with timer.track("train/epoch_total"):
            steps_per_epoch = max(len(loader) for loader in task_loaders.values())

            balanced_tasks = training_tasks * (steps_per_epoch // len(training_tasks) + 1)
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
                fake_A_ = fake_A_buffers[task].push_and_pop(fake_A)
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
                fake_B_ = fake_B_buffers[task].push_and_pop(fake_B)
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
            with timer.track("fid/compute"):
                fid_inference(epoch, opt, transforms_, task2id, Tensor, G_AB, G_BA, fid_image_dir_A, fid_image_dir_B)
                    
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
            with timer.track("sample_images/compute"):
                sample_images(epoch, opt, training_tasks, G_AB, G_BA, task2id, image_folder, Tensor)
                sample_images(epoch, opt, training_tasks, G_AB, G_BA, task2id, image_folder, Tensor, 42)
                plot_losses(logger, out_path=loss_plot_path, smooth_alpha=0.1, last_n=None, show=False)      

        
        # Copy model folder to drive
        if opt.save_model and os.path.abspath(opt.session_folder) != os.path.abspath(base_folder):
            destination = os.path.join(base_folder, "saved_models", task_name + suffix, f"model_{start_time_str}")
            copy_missing(session_model_folder, destination)
        
        # Save model checkpoints
        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            save_model_checkpoints(checkpoint_folder, epoch, G_AB, G_BA, D_A, D_B)

    # ----------------------------------
    # Save final model and run summary
    # ----------------------------------
           
    # Save training summmary
    end_time = datetime.datetime.now()
    summary = {
        "run_started_at": start_time.isoformat(timespec="seconds"),
        "run_ended_at": end_time.isoformat(timespec="seconds"),
        "total_seconds": (end_time - start_time).total_seconds(),
        "tasks": training_tasks,
        "lora": opt.lora is not None,
        "parameter_counts": param_summary,
        "timings": timer.as_dict(),
    }

    save_run_summary(local_model_folder, summary)
    # save_run_summary(session_model_folder, summary)
    # optional: also save next to local_model_folder
    save_final_models(local_model_folder, G_AB, G_BA, D_A, D_B, opt)

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()