# config.py

import argparse
import os


def get_parser(base_folder: str, data_root: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=4, help="number of epochs of training")
    parser.add_argument(
        "--tasks", 
        nargs="+",
        default=["day2night", "horse2zebra", "summer2winter_yosemite"],
        help="List of tasks/datasets to train on (space-separated)",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--lr_G", type=float, default=0.0002, help="generator learning rate")
    parser.add_argument("--lr_D", type=float, default=0.00005, help="discriminator learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=1, help="epoch from which to start lr decay")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_height", type=int, default=64, help="size of image height")
    parser.add_argument("--img_width", type=int, default=64, help="size of image width")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=1, help="interval between saving generator outputs")
    parser.add_argument("--fid_interval", type=int, default=3, help="interval between fid calculation")
    parser.add_argument("--checkpoint_interval", type=int, default=5, help="interval between saving model checkpoints")
    parser.add_argument("--n_residual_blocks", type=int, default=3, help="number of residual blocks in generator")
    parser.add_argument("--lambda_cyc", type=float, default=5.0, help="cycle loss weight")
    parser.add_argument("--lambda_id", type=float, default=0.01, help="identity loss weight")
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
    parser.add_argument(
        "--lora",
        nargs="+",
        default=None,
        help="fine tune using LoRA adapters on specified tasks. e.g. --lora day2night"
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default=None,
        help="pretrained model to finetune lora from. i.e saved_models/day2night/model_20260128_122440"
    )
    parser.add_argument("--seed", type=int, default=13)

    return parser


def parse_args(base_folder: str, data_root: str):
    parser = get_parser(base_folder, data_root)
    return parser.parse_args()
