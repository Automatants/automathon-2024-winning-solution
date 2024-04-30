import pytorch_lightning as pl
import torch.nn as nn
import torch
import argparse
import yaml
import munch
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
import torchinfo


class Imagedataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class DoubleLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(DoubleLinear, self).__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(out_features, out_features)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


class Baseline(pl.LightningModule):
    def __init__(self, config):
        super(Baseline, self).__init__()
        self.config = config
        self.model = DoubleLinear(1, config.hidden_dim)
        self.loss = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        wandb.log({"train loss": loss})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.lr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default="../../configs/baseline/config.yaml")

    args = parser.parse_args()
    with open(args.config_path, 'r') as f:
        yamlfile = munch.munchify(yaml.safe_load(f))
    config = yamlfile.config

    x = torch.randn(100, 1)
    y = 3*x + 2 + torch.randn(100, 1)

    wandb.init(project="Deepfake challenge", config=config, group=yamlfile.name, entity="automathon")

    train_dataset = Imagedataset(x, y)
    val_dataset = Imagedataset(x, y)

    model = Baseline(config)

    checkpoint_callback = ModelCheckpoint(dirpath="../../checkpoints/baseline/", every_n_train_steps=2, save_top_k=1, save_last=True,
                                 monitor="val loss", mode="min")
    checkpoint_callback.CHECKPOINT_NAME_LAST = yamlfile.name

    trainer = pl.Trainer(max_epochs=config.epoch,
                         accelerator="auto",
                         precision='16-mixed',
                         callbacks=[checkpoint_callback],)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size)

    trainer.fit(model, train_loader, val_loader)


