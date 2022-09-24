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
    # -------------------------- data settings --------------------------

    parser.add_argument(
        "--dataset_root",
        type=expandpath,
        required=True,
        default="/cluster/scratch/jehrat/ai4good/splits_jakarta/satellite",
        help="Path to dataset",
    )

    parser.add_argument(
        "--save_root",
        type=expandpath,
        required=True,
        default="/cluster/scratch/$USER/ai4good/results/",
        help="Path where to save images",
    )

    parser.add_argument(
        "--segnet_checkpoint",
        type=str,
        required=True,
        help="Path to segnet ckpt file.",
    )

    cfg = parser.parse_args()

    print(json.dumps(cfg.__dict__, indent=4, sort_keys=True))

    return cfg