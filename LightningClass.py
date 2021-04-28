from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import wandb

class Lightning(pl.LightningModule):

      def __init__(self):
            super().__init__()

            # --------------------------
            # Models
            # --------------------------

            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

      def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            output = F.log_softmax(x, dim=1)
            return output

      # --------------------------
      # Training settings
      # --------------------------

      @staticmethod
      def add_model_specific_args():
            parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
            parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                              help='input batch size for training (default: 64)')
            parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                              help='input batch size for testing (default: 1000)')
            parser.add_argument('--epochs', type=int, default=14, metavar='N',
                              help='number of epochs to train (default: 14)')
            parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                              help='learning rate (default: 1.0)')
            parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                              help='Learning rate step gamma (default: 0.7)')
            parser.add_argument('--no-cuda', action='store_true', default=False,
                              help='disables CUDA training')
            parser.add_argument('--dry-run', action='store_true', default=False,
                              help='quickly check a single pass')
            parser.add_argument('--seed', type=int, default=1, metavar='S',
                              help='random seed (default: 1)')
            parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                              help='how many batches to wait before logging training status')
            parser.add_argument('--save-model', action='store_true', default=False,
                              help='For Saving the current Model')
            args = parser.parse_args()

            return args

      def losss_func(self, output, target):
            return F.nll_loss(output, target)

      # --------------------------
      # Train & ValidationLoop
      # --------------------------

      def training_step(self, train_batch, batch_idx):
            data, target = train_batch
            output = self.forward(data)
            loss = self.losss_func(output, target)
            self.log('train_loss', loss.item())
            wandb.log({"loss": loss})
            return {'loss': loss}

      def validation_step(self, batch, batch_idx):
            data, target = batch
            output = self.forward(data)
            loss = self.losss_func(output, target)
            self.log('val_loss', loss)
            return {'val_loss': loss}

      # --------------------------
      # Optimizer
      # --------------------------

      def configure_optimizers(self):
            args = self.add_model_specific_args()
            return optim.Adam(self.parameters(), lr=args.lr)

      # --------------------------
      # Split Dataset
      # --------------------------

      def transform(self):
            return transforms.Compose([
                  transforms.ToTensor(),
                  transforms.Normalize((0.1307,), (0.3081,))
            ])

      # --------------------------
      # Init Dataloader from MNIST Dataset
      # --------------------------

      def train_dataloader(self):
            train_kwargs = {'batch_size': self.add_model_specific_args().batch_size}
            train_loader = DataLoader(
                  datasets.MNIST('../data', train=True, download=True,
                              transform=self.transform())
                  , **train_kwargs)
            return train_loader

      def val_dataloader(self):
            test_kwargs = {'batch_size': self.add_model_specific_args().test_batch_size}
            train_loader = DataLoader(
                  datasets.MNIST('../data', train=False,
                              transform=self.transform())
                  , **test_kwargs)
            return train_loader

      def validation_epoch_end(self , output):
            print(output)

def main():

      model = Lightning()
      args = model.add_model_specific_args()

      # 1. Start a W&B run
      wandb.init(project='Axon', entity='vladgesin')

      # 2. Save model inputs and hyperparameters
      config = wandb.config
      config.learning_rate = args.lr

      wandb.watch(model)
      trainer = pl.Trainer.from_argparse_args(args)
      trainer.fit(model)


if __name__ == '__main__':
    main()
