from transformers import ViTImageProcessor, ViTForImageClassification

from src.config import PretrainedConfig, DatasetConfig
from src.dataset.bicycle import create_dataloaders
from src.models.base import PretrainedImageClassifier
from src.train import train_model


if __name__ == "__main__":
    model_config = PretrainedConfig(
        model_name="google/vit-base-patch16-224",
        model_class=ViTForImageClassification,
        processor_class=ViTImageProcessor,
        num_labels=2,
        learning_rate=1e-5,
        freeze_backbone=True,
        backbone_attr="vit",
        classifier_attr="classifier",
    )
    model = PretrainedImageClassifier(model_config)

    dataset_config = DatasetConfig(max_images=1000, neg_ratio=1.0, batch_size=32)

    train_loader, val_loader = create_dataloaders(
        processor=model.processor, config=dataset_config
    )

    trained_model = train_model(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=3,
    )
