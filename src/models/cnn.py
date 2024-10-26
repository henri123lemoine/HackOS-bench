# Code taken from my COMP551 mini-project 2

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.base import Model

logger = logging.getLogger(__name__)

np.random.seed(0)


def accuracy(y_true, y_pred):
    if isinstance(y_true, torch.Tensor) and isinstance(y_pred, torch.Tensor):
        return (y_pred == y_true).float().mean().item()
    else:
        return np.mean(y_true == y_pred)


class CNN(Model):
    def __init__(
        self,
        num_channels: int = 3,
        image_size: int = 32,
        num_classes: int = 10,
        optimizer=optim.Adam,
        loss_function=nn.CrossEntropyLoss,
        lr: float = 0.001,
        **kwargs,
    ):
        super(CNN, self).__init__()

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(num_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Calculate the size of the image after the convolutional layers
        self.image_size_after_conv = image_size // 4  # Two max pooling layers

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * self.image_size_after_conv**2, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

        self.optimizer = optimizer(self.parameters(), lr=lr)
        self.loss_function = loss_function()

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
