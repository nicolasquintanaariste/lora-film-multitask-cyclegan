# config.py

import argparse
import os


def get_parser(base_folder: str, data_root: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_epochs", type=int, default=4)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["day2night", "horse2zebra", "summer2winter_yosemite"],
    )

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr_G", type=float, default=0.0002)
    parser.add_argument("--lr_D", type=float, default=0.00005)
    parser.add_argument("--b1", type=float, default=0.5)
    parser.add_argument("--b2", type=float, default=0.999)
    parser.add_argument("--decay_epoch", type=int, default=1)
    parser.add_argument("--n_cpu", type=int, default=0)

    parser.add_argument("--img_height", type=int, default=64)
    parser.add_argument("--img_width", type=int, default=64)
    parser.add_argument("--channels", type=int, default=3)

    parser.add_argument("--sample_interval", type=int, default=1)
    parser.add_argument("--fid_interval", type=int, default=3)
    parser.add_argument("--checkpoint_interval", type=int, default=2)

    parser.add_argument("--n_residual_blocks", type=int, default=3)
    parser.add_argument("--lambda_cyc", type=float, default=5.0)
    parser.add_argument("--lambda_id", type=float, default=0.01)
    parser.add_argument("--epoch", type=int, default=0)

    parser.add_argument(
        "--data_folder",
        type=str,
        default=data_root,
    )

    parser.add_argument(
        "--session_folder",
        type=str,
        default=base_folder,
    )

    parser.add_argument("--save_model", action="store_true")

    parser.add_argument("--checkpoint_model", type=str, default=None)

    parser.add_argument("--lora", nargs="+", default=None)

    parser.add_argument("--pretrained_model", type=str, default=None)

    parser.add_argument("--seed", type=int, default=13)

    return parser


def parse_args(base_folder: str, data_root: str):
    parser = get_parser(base_folder, data_root)
    return parser.parse_args()
