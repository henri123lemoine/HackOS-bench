# -*- coding: utf-8 -*-
"""HackOS.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1zv3cp3-fRmP2q2-55nol01hq3GGfztGo
"""

# Ensure necessary libraries are installed in Colab
from io import BytesIO

import numpy as np
import requests
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm

from src.dataset.url_list import LIST_OF_BICYCLES, LIST_OF_NON_BICYCLES
from src.eval import evaluate_model
from src.train import train_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class URLImageDataset(Dataset):
    def __init__(self, image_urls, label, transform=None):
        self.image_urls = image_urls
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.image_urls)

    def __getitem__(self, idx):
        url = self.image_urls[idx]
        try:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content)).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, self.label
        except Exception as e:
            print(f"Error loading image: {e}")
            return None, self.label


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

bicycle_dataset = URLImageDataset(LIST_OF_BICYCLES, label=1, transform=transform)
non_bicycle_dataset = URLImageDataset(
    LIST_OF_NON_BICYCLES, label=0, transform=transform
)
full_dataset = bicycle_dataset + non_bicycle_dataset


""" Train-test split"""

import numpy as np
from torch.utils.data import Subset

dataset_size = len(full_dataset)

indices = list(range(dataset_size))

np.random.seed(42)
np.random.shuffle(indices)

# For 60% train, 20% validation, 20% test
train_split = int(0.6 * dataset_size)
val_split = int(0.8 * dataset_size)

train_indices = indices[:train_split]
val_indices = indices[train_split:val_split]
test_indices = indices[val_split:]

train_dataset = Subset(full_dataset, train_indices)
val_dataset = Subset(full_dataset, val_indices)
test_dataset = Subset(full_dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


""" ResNet """

# Load pretrained ResNet model
resnet18 = models.resnet18(pretrained=True)
resnet18.fc = nn.Linear(
    resnet18.fc.in_features, 2
)  # Modify the final layer for binary classification
model = resnet18.to("cuda" if torch.cuda.is_available() else "cpu")

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet18.parameters(), lr=0.001)

# Train the model
train_model(resnet18, train_loader, val_loader, criterion, optimizer, num_epochs=5)
accuracy = evaluate_model(resnet18, test_loader)

""" Efficient Net"""

import torchvision.models as models

model = models.efficientnet_b7(pretrained=True)
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features, 2
)  # Modify for binary classification
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5)

accuracy = evaluate_model(model, test_loader)

""" MobileNetV2 """

mobile_netv2 = models.mobilenet_v2(pretrained=True)
mobile_netv2.classifier[1] = nn.Linear(
    mobile_netv2.classifier[1].in_features, 2
)  # Modify for binary classification
mobile_netv2 = mobile_netv2.to("cuda" if torch.cuda.is_available() else "cpu")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mobile_netv2.parameters(), lr=0.001)

train_model(mobile_netv2, train_loader, val_loader, criterion, optimizer, num_epochs=5)

accuracy = evaluate_model(mobile_netv2, test_loader)
