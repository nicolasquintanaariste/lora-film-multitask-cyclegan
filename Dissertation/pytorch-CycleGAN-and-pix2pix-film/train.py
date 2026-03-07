"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""

import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.util import init_ddp, cleanup_ddp

# Added from my version
from utils import *
from save_utils import *
from metrics_utils import *
from loss_utils import LossLogger, plot_losses


if __name__ == "__main__":
    opt = TrainOptions().parse()  # get training options
    opt.device = init_ddp()
    # dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    
    # Multitask datasets
    if "film" in opt.netG.lower():
        opt.dataset_mode = "film"
    task2id = {t: i for i, t in enumerate(opt.tasks)}
    task_datasets = {}
    for task in opt.tasks:
        task_datasets[task2id[task]] = create_dataset(opt, task) ### Change opt so that films makes use of FiLM datasets
    max_dataset_size = max(len(dataset) for dataset in task_datasets.values())  # get the number of images in the dataset.
    min_dataset_size = min(len(dataset) for dataset in task_datasets.values())
    max_iters_per_epoch = min_dataset_size // opt.batch_size
    print(f"The number of training images = {min_dataset_size}")
    tid = random.randint(0, len(opt.tasks)-1)
    
    ####################################
    # Added from my model
    ####################################
    image_folder = f"results/{opt.name}/images"
    local_model_folder = f"results/{opt.name}"
    os.makedirs(local_model_folder, exist_ok=True)
    os.makedirs(image_folder, exist_ok=True)
    loss_csv = f"{local_model_folder}/loss_log.csv"
    loss_plot_path = f"{local_model_folder}/loss_plot.png"
    logger = LossLogger(csv_path=loss_csv)
    
    start_time = datetime.datetime.now()
    timer = PhaseTimer(use_cuda_sync=True)
    transforms_ = [   # Image transformations
        # transforms.Resize(int(opt.img_height * 1.12), Image.BICUBIC), # x1.12 would make img bigger and crop edges
        transforms.Resize(256),  # Resize shortest side to img_height, maintains aspect ratio
        transforms.RandomCrop((256, 256)),  # Now crop to square
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
  
    with timer.track("prep/fid_save_real"):
        for task in opt.tasks:
            real_dir = os.path.join(opt.dataroot_general, task, "testB_normalised")
            fid_save_real(
                in_dir=os.path.join(opt.dataroot_general, task),
                out_dir=real_dir,
                transforms_=transforms_,
                max_images=250,
                batch_size=opt.batch_size,
            )
    metric_logger = MetricLogger(csv_path=os.path.join(f"results/{opt.name}", "fid_kid.csv"))
    
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots
    total_iters = 0  # the total number of training iterations
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        with timer.track("train/epoch_total"):
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()  # timer for data loading per iteration
            epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
            visualizer.reset()
            # Set epoch for DistributedSampler
            for tid, dataset in task_datasets.items():
                if hasattr(dataset, "set_epoch"):
                    dataset.set_epoch(epoch)

            # Choose one tid to train on per epoch
            if tid+1 >= len(opt.tasks):
                tid = 0
            else: 
                tid += 1 
            
            for i, data in enumerate(task_datasets[tid]):  # inner loop within one epoch
                if i >= max_iters_per_epoch: # limit iterations to the smallest dataset
                    break
                iter_start_time = time.time()  # timer for computation per iteration
                if total_iters % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time

                total_iters += opt.batch_size
                epoch_iter += opt.batch_size
                model.set_input(data, tid)  # unpack data from dataset and apply preprocessing
                model.optimize_parameters_film()  # calculate loss functions, get gradients, update network weights

                # Log losses every batch
                losses = model.get_current_losses()
                logger.log(
                    step=total_iters,
                    loss_G=losses['G'],
                    loss_GAN=losses['adversarial'],
                    loss_cycle=losses['cycle'],
                    loss_identity=losses['idt'],
                    loss_D=losses['D'],
                    loss_D_A=losses['D_A'],
                    loss_D_B=losses['D_B'],
                    dA_real_mean=model.D_A_real_mean,
                    dA_fake_mean=model.D_A_fake_mean,
                    dB_real_mean=model.D_B_real_mean,
                    dB_fake_mean=model.D_B_fake_mean,
                )

                if total_iters % opt.display_freq == 0:  # display images on visdom and save images to a HTML file
                    save_result = total_iters % opt.update_html_freq == 0
                    model.compute_visuals()
                    visualizer.display_current_results(model.get_current_visuals(), epoch, total_iters, save_result)

                if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
                    t_comp = (time.time() - iter_start_time) / opt.batch_size
                    visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                    visualizer.plot_current_losses(total_iters, losses)

                if total_iters % opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
                    print(f"saving the latest model (epoch {epoch}, total_iters {total_iters})")
                    save_suffix = f"iter_{total_iters}" if opt.save_by_iter else "latest"
                    model.save_networks(save_suffix)

                iter_data_time = time.time()

            model.update_learning_rate()  # update learning rates at the end of every epoch

            if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
                print(f"saving the model at the end of epoch {epoch}, iters {total_iters}")
                model.save_networks("latest")
                model.save_networks(epoch)
                
            # Plot FID
            if (epoch-1) % 10 == 0: #opt.fid_interval == 0:
                with timer.track("fid/compute"):
                    for task in opt.tasks:
                        fid_image_dir_A = f"results/{opt.name}/fake/{task}/A"
                        fid_image_dir_B = f"results/{opt.name}/fake/{task}/B"                       
                        fid_inference(epoch, opt, transforms_, task, task2id, Tensor, model.netG_A, model.netG_B, fid_image_dir_A, fid_image_dir_B)

                        fake_dir = os.path.join(fid_image_dir_B, f"epoch{epoch:03d}")
                        
                        fid, kid = compute_fid_kid(real_dir, fake_dir)

                        metric_logger.log(epoch=epoch, fid=fid, kid=kid, task=task)
                        plot_fid(
                            metric_logger,
                            out_path=os.path.join(f"results/{opt.name}", f"fid.png"),
                            show=False
                        )
                        plot_kid(
                            metric_logger,
                            out_path=os.path.join(f"results/{opt.name}", f"kid.png"),
                            show=False
                        )
                    
            # Generate samples and plot losses
            if epoch % 1 == 0: #opt.sample_interval == 0:
                with timer.track("sample_images/compute"):
                    sample_images(epoch, opt, model.netG_A, model.netG_B, task2id, image_folder, Tensor)
                    sample_images(epoch, opt, model.netG_A, model.netG_B, task2id, image_folder, Tensor, 42)
                    plot_losses(logger, out_path=loss_plot_path, smooth_alpha=0.1, last_n=None, show=False)      
            
            # # Copy model folder to drive
            # destination = os.path.join(base_folder, local_model_folder)
            # copy_missing(session_model_folder, destination)
                
            

            print(f"End of epoch {epoch} / {opt.n_epochs + opt.n_epochs_decay} \t Time Taken: {time.time() - epoch_start_time:.0f} sec")

            # Save training summmary
            end_time = datetime.datetime.now()
            summary = {
                "run_started_at": start_time.isoformat(timespec="seconds"),
                "run_ended_at": end_time.isoformat(timespec="seconds"),
                "total_seconds": (end_time - start_time).total_seconds(),
                "tasks": opt.tasks,
                # "lora": opt.lora is not None,
                # "parameter_counts": param_summary,
                "timings": timer.as_dict(),
            }
            save_run_summary(local_model_folder, summary)

    cleanup_ddp()
