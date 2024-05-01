import argparse
import torch
import torch.nn as nn
from vanilla import Baseline
import yaml
import munch
import json
import pandas as pd
import os


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

    sample_submission = pd.read_csv("../../data/sample_submission.csv") # id, ...
    datasetcsv = pd.read_csv("../../data/dataset.csv")  # id, file

    id_to_file = {row['id']: row['file'] for _, row in datasetcsv.iterrows()}

    for i, row in sample_submission.iterrows():
        video_id = row['id']
        file = id_to_file[video_id]
        file = file[:-4] + ".pt"
        video_path = f"../../data/processed/{file}"

        if not os.path.exists(video_path):
            continue

        faces = torch.load(video_path)

        y_hat = model(faces)
        sample_submission.loc[i, 'label'] = (y_hat > 0.5).float().item()