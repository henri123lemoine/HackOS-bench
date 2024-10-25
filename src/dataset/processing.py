from typing import Any, Callable

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from src.dataset.load_dataset import get_bicycle_dataset


def identity_transform(x: Any) -> Any:
    """Default transform function that returns input unchanged"""
    return x


def ensure_rgb(image: Any) -> np.ndarray:
    """Convert image to RGB format if needed"""
    if isinstance(image, np.ndarray):
        # If grayscale, convert to RGB
        if len(image.shape) == 2:
            return np.stack([image] * 3, axis=-1)
        # If RGB or RGBA, ensure RGB
        elif len(image.shape) == 3:
            if image.shape[-1] == 4:  # RGBA
                return image[..., :3]
            elif image.shape[-1] == 3:  # RGB
                return image
            elif image.shape[-1] == 1:  # Grayscale with channel dimension
                return np.repeat(image, 3, axis=-1)
    elif isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))

    raise ValueError(
        f"Unsupported image format: {type(image)}, shape: {getattr(image, 'shape', None)}"
    )


class PreprocessedImageDataset(Dataset):
    """Generic dataset that applies preprocessing to images"""

    def __init__(
        self,
        processor: Any,
        split: str = "train",
        neg_ratio: float = 1.0,
        max_images: int = 1000,
        random_seed: int = 42,
        transform_fn: Callable | None = None,
    ):
        train_dataset, val_dataset = get_bicycle_dataset(
            neg_ratio=neg_ratio,
            max_images=max_images,
            random_seed=random_seed,
        )

        self.base_dataset = train_dataset if split == "train" else val_dataset
        self.processor = processor
        self.transform_fn = (
            transform_fn if transform_fn is not None else identity_transform
        )

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> tuple[dict, int]:
        image, label = self.base_dataset[idx]

        # Convert to RGB if needed
        image = ensure_rgb(image)

        # Apply any additional transformations
        image = self.transform_fn(image)

        # Process image using model's processor
        processed = self.processor(images=image, return_tensors="pt")
        processed = {k: v.squeeze(0) for k, v in processed.items()}

        return processed, label


def create_dataloaders(
    processor: Any,
    batch_size: int = 16,
    neg_ratio: float = 1.0,
    max_images: int = 1000,
    random_seed: int = 42,
    num_workers: int = 4,
    transform_fn: Callable | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders"""
    train_dataset = PreprocessedImageDataset(
        processor,
        split="train",
        neg_ratio=neg_ratio,
        max_images=max_images,
        random_seed=random_seed,
        transform_fn=transform_fn,
    )

    val_dataset = PreprocessedImageDataset(
        processor,
        split="val",
        neg_ratio=neg_ratio,
        max_images=max_images,
        random_seed=random_seed,
        transform_fn=transform_fn,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader
