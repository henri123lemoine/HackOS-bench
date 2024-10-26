from dataclasses import dataclass
from typing import Type
from transformers import PreTrainedModel, PreTrainedTokenizerBase


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
    backbone_attr: str = "base_model"  # Attribute name for the backbone (e.g., 'vit' for ViT)
    classifier_attr: str = "classifier"  # Attribute name for the classification head
