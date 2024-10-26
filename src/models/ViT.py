from transformers import ViTImageProcessor, ViTForImageClassification

from src.config import PretrainedConfig
from src.models.base import PretrainedImageClassifier


def create_vit_classifier() -> PretrainedImageClassifier:
    config = PretrainedConfig(
        model_name="google/vit-base-patch16-224",
        model_class=ViTForImageClassification,
        processor_class=ViTImageProcessor,
        num_labels=2,
        learning_rate=1e-5,
        freeze_backbone=True,
        backbone_attr="vit",
        classifier_attr="classifier",
    )

    return PretrainedImageClassifier(config)


if __name__ == "__main__":
    vit = create_vit_classifier()
    print(vit.config)
