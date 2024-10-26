import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.config import PretrainedConfig
from src.settings import CACHE_PATH


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str):
        model = cls()
        model.load_state_dict(torch.load(path))
        return model

    def save_complete_model(
        self,
        file_name: str | None = None,
        dir_path: Path = CACHE_PATH,
        ext: str = "pth",
    ):
        if file_name is None:
            file_name = self.__class__.__name__
        file_path = dir_path / f"{file_name}_cls.{ext}"
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_complete_model(
        cls, file_name: str | None = None, dir_path: Path = CACHE_PATH, ext: str = "pth"
    ):
        if file_name is None:
            file_name = cls.__name__
        file_path = dir_path / f"{file_name}_cls.{ext}"
        with open(file_path, "rb") as f:
            return pickle.load(f)

    def fit(self, X: Any, y: Any):
        raise NotImplementedError("Subclasses must implement fit method")

    def predict(self, X: Any) -> Any:
        raise NotImplementedError("Subclasses must implement predict method")

    def evaluate(self, X: Any, y: Any) -> dict[str, float]:
        raise NotImplementedError("Subclasses must implement evaluate method")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement forward method")


class PretrainedImageClassifier(Model):
    """Generic classifier that can work with any pretrained model"""

    def __init__(
        self,
        config: PretrainedConfig,
    ):
        super().__init__()
        self.config = config

        # Initialize model and processor
        self.processor = config.processor_class.from_pretrained(config.model_name)

        ## NOTE ##
        # ImageNet:
        ## 444	bicycle-built-for-two, tandem bicycle, tandem
        ## 671	mountain bike, all-terrain bike, off-roader
        ## 670	motor scooter, scooter (to contrast with?)

        # First load the model with original classification head
        original_model = config.model_class.from_pretrained(config.model_name)

        # Get bicycle-related weights from the original classifier
        original_classifier = getattr(original_model, config.classifier_attr)
        if hasattr(original_classifier, "weight"):
            # For binary classification: [bicycle, background]
            bicycle_weights = original_classifier.weight[
                [671, 444]
            ]  # mountain bike and tandem
            bicycle_weight = bicycle_weights.mean(dim=0, keepdim=True)  # average them

            # For background, average everything except bicycles and similar vehicles
            exclude_indices = {670, 671, 444}  # exclude scooter and bicycle classes
            background_indices = [
                i
                for i in range(original_classifier.weight.size(0))
                if i not in exclude_indices
            ]
            background_weight = original_classifier.weight[background_indices].mean(
                dim=0, keepdim=True
            )

            # Put bicycle first to match dataset labels
            initial_weights = torch.cat([bicycle_weight, background_weight], dim=0)

            if hasattr(original_classifier, "bias"):
                bicycle_biases = original_classifier.bias[[671, 444]]
                bicycle_bias = bicycle_biases.mean().unsqueeze(0)
                background_bias = (
                    original_classifier.bias[background_indices].mean().unsqueeze(0)
                )
                initial_bias = torch.cat([bicycle_bias, background_bias], dim=0)

        # Now create our binary classification model
        self.model = config.model_class.from_pretrained(
            config.model_name,
            num_labels=config.num_labels,
            ignore_mismatched_sizes=True,
        )

        # Initialize the new classifier with bicycle-related weights
        new_classifier = getattr(self.model, config.classifier_attr)
        if hasattr(new_classifier, "weight") and "initial_weights" in locals():
            print(
                "Initializing classifier with pretrained bicycle weights"
            )  # Debug print
            with torch.no_grad():
                new_classifier.weight.copy_(initial_weights)
                if hasattr(new_classifier, "bias") and "initial_bias" in locals():
                    new_classifier.bias.copy_(initial_bias)

        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, pixel_values=None):
        if isinstance(x, np.ndarray):
            if len(x.shape) == 3:
                x = [x]
            processed = self.processor(images=x, return_tensors="pt")
            return self.model(**processed).logits
        return self.model(x).logits

    def predict(self, x: np.ndarray) -> int:
        self.eval()
        with torch.no_grad():
            outputs = self(x)
            _, predicted = torch.max(outputs, 1)
            return predicted.item() if len(outputs) == 1 else predicted.numpy()
