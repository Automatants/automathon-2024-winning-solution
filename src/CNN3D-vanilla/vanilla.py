import pytorch_lightning as pl
import torch.nn as nn
import torch
import argparse
import yaml
import munch
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
import torchinfo
import os
import json


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, config, metadata_path):
        self.folder_path = "../../data/processed"
        metadata = json.load(open(metadata_path))
        self.filename = [f"{k[:-4]}.pt" for k in metadata.keys()]

        # remove files that do not exist
        filelist = []
        for f in self.filename:
            if os.path.exists(f"{self.folder_path}/{f}"):
                filelist.append(f)

        self.filename = filelist

        self.labels = [1 if metadata[v] == 'fake' else 0 for v in metadata.keys()]

        print(sum(self.labels)/len(self.labels))

        self.config = config

    def __len__(self):
        return len(self.filename)

    def __getitem__(self, idx):
        filename = self.filename[idx]
        x = torch.load(f'{self.folder_path}/{filename}')
        start_middle = x.shape[1] // 2 - self.config.n_frames // 2
        x = x[:, start_middle:start_middle + self.config.n_frames]
        y = self.labels[idx]

        if x.shape[1] != self.config.n_frames:
            print(x.shape[1], filename)
            return self.__getitem__((idx+1) % len(self))

        return x.float()/255.0, y


class CNN3d(nn.Module):
    def __init__(self, channel_list):
        super().__init__()

        self.conv1 = nn.Conv3d(channel_list[0], channel_list[1], kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(channel_list[1], channel_list[2], kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(channel_list[2], channel_list[3], kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(channel_list[3], channel_list[4], kernel_size=3, padding=1)
        self.conv5 = nn.Conv3d(channel_list[4], channel_list[5], kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(2)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm3d(channel_list[1])
        self.bn2 = nn.BatchNorm3d(channel_list[2])
        self.bn3 = nn.BatchNorm3d(channel_list[3])
        self.bn4 = nn.BatchNorm3d(channel_list[4])

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.relu(x)
        x = self.pool(x)
        x = self.bn2(self.conv2(x))
        x = self.relu(x)
        x = self.pool(x)
        x = self.bn3(self.conv3(x))
        x = self.relu(x)
        x = self.pool(x)
        x = self.bn4(self.conv4(x))
        x = self.relu(x)
        x = self.pool(x)
        x = self.relu(self.conv5(x))
        print(x.shape)
        return x


class PredictionHead(nn.Module):
    def __init__(self, in_features):
        super(PredictionHead, self).__init__()
        self.linear1 = nn.Linear(in_features, 256)
        self.linear2 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x.squeeze()


class Baseline(pl.LightningModule):
    def __init__(self, config):
        super(Baseline, self).__init__()
        self.config = config
        self.model = CNN3d(config.channels)
        self.head = PredictionHead(config.channels[-1] * 8 * 8)
        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(0.3))

    def forward(self, x):
        return self.head(self.model(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y.float())

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y.float())

        self.log("val_loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.lr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default="../../configs/CNN3D-vanilla/config.yaml")

    args = parser.parse_args()
    with open(args.config_path, 'r') as f:
        yamlfile = munch.munchify(yaml.safe_load(f))
    config = yamlfile.config


    # model = CNN3d(config.channels)
    # torchinfo.summary(model, (1, 3, config.n_frames, 128, 128))

    logger = pl.loggers.WandbLogger(project="Deepfake challenge", config=config, group=yamlfile.name, entity="automathon")

    model = Baseline(config)

    checkpoint_callback = ModelCheckpoint(dirpath="../../checkpoints/CNN3D-vanilla/", every_n_train_steps=2, save_top_k=1, save_last=True,
                                          monitor="val_loss", mode="min")
    checkpoint_callback.CHECKPOINT_NAME_LAST = yamlfile.name

    trainer = pl.Trainer(max_epochs=config.epoch,
                         accelerator="auto",
                         precision='16-mixed',
                         callbacks=[checkpoint_callback],
                         logger=logger,)

    train_dataset = VideoDataset(config, "../../data/raw/metadata.json")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)

    trainer.fit(model, train_loader)


