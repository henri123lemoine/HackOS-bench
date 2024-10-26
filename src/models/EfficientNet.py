import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import requests
from PIL import Image
from io import BytesIO
from tqdm.notebook import tqdm
import torchvision.models as models

# Load pretrained EfficientNet model
model = models.efficientnet_b7(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)  # Modify for binary classification
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, epochs=5)