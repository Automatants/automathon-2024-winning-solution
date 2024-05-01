import pytorch_lightning as pl
import torch.nn as nn
import torch
import argparse
import yaml
import munch
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
import torchinfo


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

        timestamp = int(np.random.choice(range(x.shape[1])))

        x = x[:, timestamp]
        y = self.labels[idx]
        return x, y

class CNN2d(nn.Module):
    def __init__(self, channel_list):
        super().__init__()

        self.conv1 = nn.Conv2d(channel_list[0], channel_list[1], kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channel_list[1], channel_list[2], kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(channel_list[2], channel_list[3], kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(channel_list[3], channel_list[4], kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(channel_list[4], channel_list[5], kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(channel_list[5], channel_list[6], kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(channel_list[6], channel_list[7], kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(channel_list[7], channel_list[8], kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(channel_list[8], channel_list[9], kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(channel_list[1])
        self.bn2 = nn.BatchNorm2d(channel_list[2])
        self.bn3 = nn.BatchNorm2d(channel_list[3])
        self.bn4 = nn.BatchNorm2d(channel_list[4])
        self.bn5 = nn.BatchNorm2d(channel_list[5])
        self.bn6 = nn.BatchNorm2d(channel_list[6])
        self.bn7 = nn.BatchNorm2d(channel_list[7])
        self.bn8 = nn.BatchNorm2d(channel_list[8])
        self.bn9 = nn.BatchNorm2d(channel_list[9])

        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x1 = self.pool(x) #64
        x1 = self.dropout(x1)

        x = self.relu(self.conv2(x1))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x + x1))
        x = self.pool(x) #32
        x = self.dropout(x)

        x = self.relu(self.conv5(x))
        x2 = self.pool(x) #16
        x2 = self.dropout(x2)

        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x))
        x = self.relu(self.conv8(x + x2))
        x = self.pool(x) #8
        x = self.dropout(x)

        x = self.relu(self.conv9(x))
        x = self.pool(x) #4
        x = self.dropout(x)
        return x


class PredictionHead(nn.Module):
    def __init__(self, in_features):
        super(PredictionHead, self).__init__()
        self.linear1 = nn.Linear(in_features, 1)
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
        self.model = CNN2d(config.channels)
        self.head = PredictionHead(config.channels[-1] * 4 * 4)
        self.loss = nn.BCELoss()

    def forward(self, x):
        return self.head(self.model(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        # wandb.log({"train loss": loss})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.lr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default="../../configs/CNN2D-sugar/config.yaml")

    args = parser.parse_args()
    with open(args.config_path, 'r') as f:
        yamlfile = munch.munchify(yaml.safe_load(f))
    config = yamlfile.config


    model = CNN2d(config.channels)
    torchinfo.summary(model, (1, 3, 128, 128))

    logger = pl.loggers.WandbLogger(project="Deepfake challenge", config=config, group=yamlfile.name, entity="automathon")

    model = Baseline(config)

    checkpoint_callback = ModelCheckpoint(dirpath="../../checkpoints/CNN2D-sugar/", every_n_train_steps=2, save_top_k=1, save_last=True,
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



