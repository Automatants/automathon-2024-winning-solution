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
from torchvision.models import efficientnet_b3, efficientnet_b0
# from einops import rearrange


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, config, metadata_path):
        self.folder_path = "/raid/home/automathon_2024/account24/data/processed3"
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

        x = x.float()

        x_prime1 = torch.mean(x[:, 1, :, :] - x[:, 0, :, :], dim=0)
        x_prime2 = torch.mean(x[:, 2, :, :] - x[:, 1, :, :], dim=0)
        x_prime3 = torch.mean(x[:, 3, :, :] - x[:, 2, :, :], dim=0)

        xprime = torch.stack([x_prime1, x_prime2, x_prime3], dim=0)

        return x/255.0, xprime, y


class PredictionHead(nn.Module):
    def __init__(self):
        super(PredictionHead, self).__init__()
        self.linear1 = nn.LazyLinear(256)
        self.linear2 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x, xprime):
        print("shape", x.shape)
        x = self.flatten(x)
        xprime = self.flatten(xprime)

        x = torch.cat([x, xprime], dim=1)

        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x.squeeze()


class EfficientNetPrime(nn.Module):
    def __init__(self):
        super(EfficientNetPrime, self).__init__()
        self.model = efficientnet_b0(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-1])

    def forward(self, x):
        x = self.model(x)
        return x


class Baseline(pl.LightningModule):
    def __init__(self, config):
        super(Baseline, self).__init__()
        self.config = config

        self.model = efficientnet_b3(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-1])

        self.prime = EfficientNetPrime()

        self.head = PredictionHead()
        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(0.3))

    def forward(self, x, xprime):
        x = x[:, :, 0, :, :]
        img_features = self.model(x)
        prime_features = self.prime(xprime)
        return self.head(img_features, prime_features)

    def training_step(self, batch, batch_idx):
        x, xprime, y = batch
        y_hat = self(x, xprime)
        loss = self.loss(y_hat, y.float())

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, xprime, y = batch
        y_hat = self(x, xprime)
        loss = self.loss(y_hat, y.float())

        self.log("val_loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.lr)


if __name__ == '__main__':



    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default="../../configs/DoubleCNN/config.yaml")

    args = parser.parse_args()
    with open(args.config_path, 'r') as f:
        yamlfile = munch.munchify(yaml.safe_load(f))
    config = yamlfile.config


    # model = CNN3d(config.channels)
    # torchinfo.summary(model, (1, 3, config.n_frames, 128, 128))

    logger = pl.loggers.WandbLogger(project="Deepfake challenge", config=config, group=yamlfile.name, entity="automathon")

    model = Baseline(config)

    checkpoint_callback = ModelCheckpoint(dirpath="../../checkpoints/DoubleCNN/", every_n_epochs=1, save_top_k=-1)
    checkpoint_callback.CHECKPOINT_NAME_LAST = yamlfile.name

    trainer = pl.Trainer(max_epochs=config.epoch,
                         accelerator="auto",
                         precision='16-mixed',
                         callbacks=[checkpoint_callback],
                         logger=logger,)

    train_dataset = VideoDataset(config, "/raid/home/automathon_2024/account24/data/metadata.json")
    val_dataset = VideoDataset(config, "/raid/home/automathon_2024/account24/data/metadata_info.json")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, num_workers=2)

    trainer.fit(model, train_loader, val_loader)


