from typing import Type
from pathlib import Path

import torch
from dataclasses import dataclass
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.settings import MODELS_PATH, DATASETS_PATH


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


@dataclass
class DatasetConfig:
    """Configuration for dataset creation."""

    cache_dir: Path = DATASETS_PATH
    max_images: int = 1000
    neg_ratio: float = 1.0
    train_ratio: float = 0.8
    random_seed: int = 42
    batch_size: int = 16
    num_workers: int = 4
    bicycle_label: int = 2  # COCO dataset bicycle class label
