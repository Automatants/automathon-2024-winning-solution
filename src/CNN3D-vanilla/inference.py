import argparse
import torch
import torch.nn as nn
from vanilla import Baseline
import yaml
import munch
import json


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default="../../configs/CNN3D-vanilla/config.yaml")
    parser.add_argument('--checkpoint_path', type=str)
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        yamlfile = munch.munchify(yaml.safe_load(f))
    config = yamlfile.config

    model = Baseline.load_from_checkpoint(args.checkpoint_path)
    model.eval()

    # metadata