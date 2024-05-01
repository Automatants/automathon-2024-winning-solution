import argparse
import torch
import torch.nn as nn
from mix import Baseline
import yaml
import munch
import json
import pandas as pd
import os


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default="../../configs/CNNmix/config.yaml")
    parser.add_argument('--checkpoint_path', type=str)
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        yamlfile = munch.munchify(yaml.safe_load(f))
    config = yamlfile.config

    model = Baseline.load_from_checkpoint(args.checkpoint_path, config=config)
    model.eval()
    model.to(device)

    sample_submission = pd.read_csv("../../data/sample_submission.csv") # id, ...
    datasetcsv = pd.read_csv("../../data/dataset.csv")  # id, file

    id_to_file = {row['id']: row['file'] for _, row in datasetcsv.iterrows()}

    for i, row in sample_submission.iterrows():
        video_id = row['id']
        file = id_to_file[video_id]
        file = file[:-4] + ".pt"
        video_path = f"../../data/processed/{file}"

        if not os.path.exists(video_path):
            sample_submission.loc[i, 'label'] = 1
            continue

        faces = torch.load(video_path)
        start_middle = faces.size(1) // 2 - config.n_frames//2
        faces = faces[:, start_middle:start_middle+config.n_frames]
        faces = faces.unsqueeze(0).half().to(device) / 255.0

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                y_hat = model(faces)

        sample_submission.loc[i, 'label'] = y_hat.item()
        print(f"{i}, {video_id}, {y_hat.item()}")

    sample_submission.to_csv("submission.csv", index=False)