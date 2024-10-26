import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.models.base import PretrainedImageClassifier
from src.models.ViT import create_vit_classifier
from src.dataset.processing import create_dataloaders
from src.settings import MODELS_PATH

def train_model(
    model: PretrainedImageClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_path: str = "best_model.pth",
) -> PretrainedImageClassifier:
    """Generic training function for any pretrained model"""
    model = model.to(device)
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch, labels in progress_bar:
            labels = labels.to(device)
            batch = {k: v.to(device) for k, v in batch.items()}

            model.optimizer.zero_grad()
            outputs = model.model(**batch).logits
            loss = model.criterion(outputs, labels)
            loss.backward()
            model.optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            progress_bar.set_postfix(
                {
                    "loss": f"{train_loss/train_total:.4f}",
                    "acc": f"{train_correct/train_total:.4f}",
                }
            )

        # Validation phase
        val_metrics = validate_model(model, val_loader, device)

        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Train Acc: {train_correct/train_total:.4f}")
        print(f"Val Acc: {val_metrics['accuracy']:.4f}")

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save(model.state_dict(), save_path)

    return model


def validate_model(
    model: PretrainedImageClassifier,
    val_loader: DataLoader,
    device: str,
) -> dict:
    """Validate model performance"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch, labels in val_loader:
            labels = labels.to(device)
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model.model(**batch).logits
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return {
        "accuracy": correct / total if total > 0 else 0,
        "total_samples": total,
    }


if __name__ == "__main__":
    model = create_vit_classifier()

    train_loader, val_loader = create_dataloaders(
        processor=model.processor,
        batch_size=16,  # Adjust based on your GPU memory
        max_images=3000,  # Can increase this for more training data
    )

    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=5,
        save_path=MODELS_PATH / "best_vit_model.pth",
    )
