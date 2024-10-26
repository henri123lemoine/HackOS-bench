import numpy as np
import torch
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


def test_model_forward(model: PretrainedImageClassifier) -> bool:
    """Test if model can process a single image"""
    try:
        # Create dummy input
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        # Test forward pass
        with torch.no_grad():
            output = model.predict(dummy_image)

        # Check output
        assert isinstance(output, (int, np.ndarray)), "Invalid output type"
        print("Model forward pass test successful")
        return True

    except Exception as e:
        print(f"Model forward pass test failed: {e}")
        return False


if __name__ == "__main__":
    vit = create_vit_classifier()
    test_model_forward(vit)
