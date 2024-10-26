from typing import Type
from pathlib import Path

import torch
from dataclasses import dataclass
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.settings import MODELS_PATH


@dataclass
class PretrainedConfig:
    """Configuration for pretrained models"""

    model_name: str
    model_class: Type[PreTrainedModel]
    processor_class: Type[PreTrainedTokenizerBase]
    num_labels: int = 2
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    freeze_backbone: bool = True
    # Attribute name for the backbone (e.g., 'vit' for ViT)
    backbone_attr: str = "base_model"
    classifier_attr: str = "classifier"  # Attribute name for the classification head
    save_path: str | Path = MODELS_PATH
    checkpoint_name: str = "best_model.pth"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
