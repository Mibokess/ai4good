import argparse
import os
import json


def expandpath(path):
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def command_line_parser():
    parser = argparse.ArgumentParser(
        add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # -------------------------- wandb settings --------------------------
    parser.add_argument(
        "--project",
        type=str,
        default="Land Use Mapping CycleGAN",
        help="Name for your run to wandb project.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="AI4Good",
        help="Name for your run to easier identify it.",
    )

    # -------------------------- logging settings --------------------------
    parser.add_argument(
        "--log_dir",
        type=expandpath,
        default="/cluster/scratch/$USER/logs",
        help="Place for artifacts and logs",
    )
    parser.add_argument(
        "--use_wandb", type=str2bool, default=True, help="Use WandB for logging"
    )

    # -------------------------- training settings --------------------------
    parser.add_argument(
        "--num_epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of samples in a batch for training",
    )
    parser.add_argument(
        "--set_size",
        type=int,
        default=1000,
        help="Size of train set",
    )
    parser.add_argument(
        "--batch_size_validation",
        type=int,
        default=4,
        help="Number of samples in a batch for validation",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume training from checkpoint, which can also be an AWS link s3://...",
    )

    # -------------------------- model settings --------------------------
    parser.add_argument(
        "--use_builing_labels",
        type=str2bool,
        default=False,
        help="Use building instead of street labels",
    )

    parser.add_argument(
        "--use_unet_light",
        type=str2bool,
        default=False,
        help="Use lightweight Unet instead of more elaborate one",
    )
    # -------------------------- data settings --------------------------
    parser.add_argument(
        "--dataset_root",
        type=expandpath,
        default="/cluster/scratch/jehrat/ai4good/splits_jakarta/satellite",
        help="Path to dataset",
    )

    parser.add_argument(
        "--domainA_dir",
        type=expandpath,
        default='mboss/splits_new',
        help="Path to domain A dataset",
    )

    parser.add_argument(
        "--domainB_dir",
        type=expandpath,
        default='jehrat/ai4good/splits_jakarta/satellite',
        help="Path to domain B dataset",
    )

    # -------------------------- hardware settings --------------------------
    parser.add_argument("--gpu", type=str2bool, default=True, help="GPU usage")
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of worker threads fetching training data",
    )
    parser.add_argument(
        "--workers_validation",
        type=int,
        default=4,
        help="Number of worker threads fetching validation data",
    )

    cfg = parser.parse_args()

    print(json.dumps(cfg.__dict__, indent=4, sort_keys=True))

    return cfg