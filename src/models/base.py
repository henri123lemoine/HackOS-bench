import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.settings import CACHE_PATH

from src.config import PretrainedConfig


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
        self.model = config.model_class.from_pretrained(
            config.model_name,
            num_labels=config.num_labels,
            ignore_mismatched_sizes=True,
        )

        # Freeze backbone if specified
        if config.freeze_backbone:
            backbone = getattr(self.model, config.backbone_attr)
            for param in backbone.parameters():
                param.requires_grad = False
            # Ensure classifier is trainable
            classifier = getattr(self.model, config.classifier_attr)
            for param in classifier.parameters():
                param.requires_grad = True

        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
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
