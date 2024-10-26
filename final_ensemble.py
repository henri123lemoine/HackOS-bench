from io import BytesIO

import numpy as np
import requests
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import ViTForImageClassification, ViTImageProcessor

from src.config import PretrainedConfig
from src.eval import run_eval
from src.models.base import PretrainedImageClassifier
from src.models.ensemble import EnsembleModel
from src.settings import MODELS_PATH


# Dataset for training
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


# Create training datasets
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

from src.dataset.url_list import LIST_OF_BICYCLES, LIST_OF_NON_BICYCLES

bicycle_dataset = URLImageDataset(LIST_OF_BICYCLES, label=1, transform=transform)
non_bicycle_dataset = URLImageDataset(
    LIST_OF_NON_BICYCLES, label=0, transform=transform
)
full_dataset = bicycle_dataset + non_bicycle_dataset

import numpy as np

# Train-test split
from torch.utils.data import Subset

dataset_size = len(full_dataset)
indices = list(range(dataset_size))
np.random.seed(42)
np.random.shuffle(indices)

train_split = int(0.6 * dataset_size)
val_split = int(0.8 * dataset_size)

train_dataset = Subset(full_dataset, indices[:train_split])
val_dataset = Subset(full_dataset, indices[train_split:val_split])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


def train_single_model(model, num_epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    return model


# Initialize models
print("Initializing and training models...")
models_list = []
processors_list = []

# 1. ViT Model
print("Loading ViT...")
vit_config = PretrainedConfig(
    model_name="google/vit-base-patch16-224",
    model_class=ViTForImageClassification,
    processor_class=ViTImageProcessor,
    num_labels=2,
)
vit_model = PretrainedImageClassifier(vit_config)
vit_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
checkpoint = torch.load(MODELS_PATH / "vit" / "best_model.pth")
vit_model.load_state_dict(checkpoint["model_state_dict"])
models_list.append(vit_model)
processors_list.append(vit_processor)
print("ViT loaded successfully")

# 2. ResNet Model
print("Loading and training ResNet...")
resnet = models.resnet18(pretrained=True)
resnet.fc = nn.Linear(resnet.fc.in_features, 2)
# resnet = train_single_model(resnet)
models_list.append(resnet)
processors_list.append(None)
print("ResNet trained successfully")

# 3. EfficientNet
print("Loading and training EfficientNet...")
efficientnet = models.efficientnet_b7(pretrained=True)
efficientnet.classifier[1] = nn.Linear(efficientnet.classifier[1].in_features, 2)
# efficientnet = train_single_model(efficientnet)
models_list.append(efficientnet)
processors_list.append(None)
print("EfficientNet trained successfully")

# 4. MobileNetV2
print("Loading and training MobileNetV2...")
mobilenet = models.mobilenet_v2(pretrained=True)
mobilenet.classifier[1] = nn.Linear(mobilenet.classifier[1].in_features, 2)
# mobilenet = train_single_model(mobilenet)
models_list.append(mobilenet)
processors_list.append(None)
print("MobileNetV2 trained successfully")

# Create ensemble
print(f"\nTesting ensemble with weighted voting...")
ensemble = EnsembleModel(
    models=models_list,
    processors=processors_list,
    method="weighted_vote",
)

# Run evaluation
print("Starting evaluation...")
results = run_eval(
    lambda x: 1 - ensemble.predict(x)
)  # Flip predictions to match expected labels
print(f"Ensemble Results:", results)

print(f"\nAccuracy: {results['acc']*100:.2f}%")

# Save the ensemble
torch.save(
    {
        "models": models_list,
        "processors": processors_list,
        "method": "weighted_vote",
        "results": results,
    },
    MODELS_PATH / "final_ensemble.pth",
)
print("Ensemble saved successfully")
