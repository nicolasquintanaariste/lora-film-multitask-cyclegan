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


if __name__ == "__main__":
    opt = TrainOptions().parse()  # get training options
    opt.device = init_ddp()
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)  # get the number of images in the dataset.
    print(f"The number of training images = {dataset_size}")
    
    ####################################
    # Added from my model
    ####################################
    base_folder = "Dissertation\\pytorch-CycleGAN-and-pix2pix-adaptation"
    image_folder = f"Dissertation\\pytorch-CycleGAN-and-pix2pix-adaptation\\results\\{opt.name}\\images"
    timer = PhaseTimer(use_cuda_sync=True)
    transforms_ = [   # Image transformations
        # transforms.Resize(int(opt.img_height * 1.12), Image.BICUBIC), # x1.12 would make img bigger and crop edges
        transforms.Resize(256),  # Resize shortest side to img_height, maintains aspect ratio
        transforms.RandomCrop((256, 256)),  # Now crop to square
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    task2id = {"horse2zebra": 0}  # this is just a placeholder
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    fid_image_dir_A = f"Dissertation\\pytorch-CycleGAN-and-pix2pix-adaptation\\results\\{opt.name}\\fake\\A"
    fid_image_dir_B = f"Dissertation\\pytorch-CycleGAN-and-pix2pix-adaptation\\results\\{opt.name}\\fake\\B"
    real_dir = os.path.join(opt.dataroot, "testB_normalised")
    with timer.track("prep/fid_save_real"):
        fid_save_real(
            in_dir=os.path.join(opt.dataroot),
            out_dir=real_dir,
            transforms_=transforms_,
            max_images=250,
            batch_size=opt.batch_size,
        )
    metric_logger = MetricLogger(csv_path=os.path.join(f"Dissertation\\pytorch-CycleGAN-and-pix2pix-adaptation\\results\\{opt.name}", "fid_kid.csv"))
    
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots
    total_iters = 0  # the total number of training iterations
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()
        # Set epoch for DistributedSampler
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:  # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, total_iters, save_result)

            if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
                losses = model.get_current_losses()
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
        if epoch % 5 == 0: #opt.fid_interval == 0:
            with timer.track("fid/compute"):
                fid_inference(epoch, opt, transforms_, task2id, Tensor, model.netG_A, model.netG_B, fid_image_dir_A, fid_image_dir_B)

                fake_dir = os.path.join(fid_image_dir_B, f"epoch{epoch:03d}")
                
                fid, kid = compute_fid_kid(real_dir, fake_dir)

                metric_logger.log(epoch=epoch + 1, fid=fid, kid=kid)
                plot_fid(
                    metric_logger,
                    out_path=os.path.join(f"Dissertation\\pytorch-CycleGAN-and-pix2pix-adaptation\\results\\{opt.name}", f"fid.png"),
                    show=False
                )
                plot_kid(
                    metric_logger,
                    out_path=os.path.join(f"Dissertation\\pytorch-CycleGAN-and-pix2pix-adaptation\\results\\{opt.name}", f"kid.png"),
                    show=False
                )
                
        # Generate samples and plot losses
        if epoch % 1 == 0: #opt.sample_interval == 0:
            with timer.track("sample_images/compute"):
                sample_images(epoch, opt, ["horse2zebra"], model.netG_A, model.netG_B, task2id, image_folder, Tensor)
                sample_images(epoch, opt, ["horse2zebra"], model.netG_A, model.netG_B, task2id, image_folder, Tensor, 42)
                #plot_losses(logger, out_path=loss_plot_path, smooth_alpha=0.1, last_n=None, show=False)      

        
        # # Copy model folder to drive
        # destination = os.path.join(base_folder, local_model_folder)
        # copy_missing(session_model_folder, destination)
            
        

        print(f"End of epoch {epoch} / {opt.n_epochs + opt.n_epochs_decay} \t Time Taken: {time.time() - epoch_start_time:.0f} sec")

    cleanup_ddp()
