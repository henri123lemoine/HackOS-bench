import torch
from transformers import ViTImageProcessor, ViTForImageClassification

from src.config import PretrainedConfig, DatasetConfig
from src.dataset.bicycle import create_dataloaders
from src.models.base import PretrainedImageClassifier
from src.train import train_model, validate_model


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

    # Print model parameter status
    for name, param in model.named_parameters():
        print(f"{name}: requires_grad = {param.requires_grad}")

    dataset_config = DatasetConfig(max_images=1000, neg_ratio=1.0, batch_size=32)

    train_loader, val_loader = create_dataloaders(
        processor=model.processor, config=dataset_config
    )

    # First validate without any training
    print("\nValidating initial model performance...")
    initial_metrics = validate_model(model, val_loader, model.config.device)
    print(f"Initial validation accuracy: {initial_metrics['accuracy']:.4f}")

    # Train with frozen backbone first
    print("\nTraining with frozen backbone...")
    trained_model = train_model(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=3,
    )

    # Optional: Unfreeze and continue training with lower learning rate
    if initial_metrics["accuracy"] > 0.6:  # Only if initial training went well
        print("\nUnfreezing backbone and continuing training...")
        backbone = getattr(trained_model.model, trained_model.config.backbone_attr)
        for param in backbone.parameters():
            param.requires_grad = True

        # Reduce learning rate for fine-tuning
        trained_model.optimizer = torch.optim.AdamW(
            [p for p in trained_model.model.parameters() if p.requires_grad],
            lr=1e-6,  # Even lower learning rate for fine-tuning
            weight_decay=0.01,
        )

        trained_model = train_model(
            trained_model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=2,
        )
