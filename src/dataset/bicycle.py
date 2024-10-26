from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Dataset as HFDataset
from PIL import Image
from tqdm.auto import tqdm

from src.config import DatasetConfig


class BicycleDataset(Dataset):
    """Dataset for bicycle classification with efficient caching and processing."""

    def __init__(
        self,
        processor: Any,
        split: str = "train",
        config: Optional[DatasetConfig] = None,
        transform_fn: Optional[Callable] = None,
    ):
        self.config = config or DatasetConfig()
        self.processor = processor
        self.transform_fn = transform_fn
        self.split = split
        self.images, self.labels = self._load_or_create_dataset()

    def _load_or_create_dataset(self) -> Tuple[list[Image.Image], list[int]]:
        """Load dataset from cache or create new one."""
        cache_file = self._get_cache_path()

        if cache_file.exists():
            try:
                cache_data = torch.load(cache_file)
                split_data = cache_data[self.split]
                self._log_stats(split_data["images"], split_data["labels"])
                return split_data["images"], split_data["labels"]
            except Exception as e:
                print(f"Cache load failed: {e}. Creating new dataset...")

        # Create new dataset if cache missing or invalid
        return self._create_new_dataset()

    def _get_cache_path(self) -> Path:
        """Get path for cached dataset."""
        cache_dir = self.config.cache_dir / "coco2017"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return (
            cache_dir / f"coco2017_{self.config.max_images}_{self.config.neg_ratio}.pt"
        )

    def _create_new_dataset(self) -> Tuple[list[Image.Image], list[int]]:
        """Create new dataset from COCO."""
        # Load raw dataset
        dataset = load_dataset("rafaelpadilla/coco2017", split="train")

        # Filter bicycle and non-bicycle images
        bicycle_dataset = dataset.filter(
            lambda x: self.config.bicycle_label in x["objects"]["label"], num_proc=4
        )
        non_bicycle_dataset = dataset.filter(
            lambda x: self.config.bicycle_label not in x["objects"]["label"], num_proc=4
        )

        # Sample balanced dataset
        images, labels = self._create_balanced_dataset(
            bicycle_dataset, non_bicycle_dataset
        )

        # Split and cache
        self._split_and_cache_dataset(images, labels)

        # Return appropriate split
        split_idx = int(len(images) * self.config.train_ratio)
        if self.split == "train":
            return images[:split_idx], labels[:split_idx]
        return images[split_idx:], labels[split_idx:]

    def _create_balanced_dataset(
        self, pos_dataset: HFDataset, neg_dataset: HFDataset
    ) -> Tuple[list[Image.Image], list[int]]:
        """Create balanced dataset of bicycle/non-bicycle images."""
        # Calculate sample sizes
        n_pos = min(
            len(pos_dataset), self.config.max_images // (1 + int(self.config.neg_ratio))
        )
        n_neg = min(int(n_pos * self.config.neg_ratio), len(neg_dataset))

        # Sample indices
        rng = np.random.RandomState(self.config.random_seed)
        pos_indices = rng.choice(len(pos_dataset), n_pos, replace=False)
        neg_indices = rng.choice(len(neg_dataset), n_neg, replace=False)

        # Load images
        pos_images = self._load_images(
            pos_dataset, pos_indices, "Loading bicycle images"
        )
        neg_images = self._load_images(
            neg_dataset, neg_indices, "Loading non-bicycle images"
        )

        # Combine and shuffle
        images = pos_images + neg_images
        labels = [1] * len(pos_images) + [0] * len(neg_images)

        # Shuffle with same seed
        combined = list(zip(images, labels))
        rng.shuffle(combined)
        images, labels = zip(*combined)

        return list(images), list(labels)

    def _load_images(
        self, dataset: HFDataset, indices: np.ndarray, desc: str
    ) -> list[Image.Image]:
        """Load and process images from dataset."""
        return [dataset[int(i)]["image"] for i in tqdm(indices, desc=desc)]

    def _split_and_cache_dataset(
        self, images: list[Image.Image], labels: list[int]
    ) -> None:
        """Split dataset and save to cache."""
        split_idx = int(len(images) * self.config.train_ratio)
        cache_data = {
            "train": {"images": images[:split_idx], "labels": labels[:split_idx]},
            "val": {"images": images[split_idx:], "labels": labels[split_idx:]},
        }
        torch.save(cache_data, self._get_cache_path())

    def _log_stats(self, images: list[Image.Image], labels: list[int]) -> None:
        """Log dataset statistics."""
        n_pos = sum(labels)
        print(f"""
        {self.split} dataset loaded:
        Total images: {len(images)}
        Positive (bicycle) images: {n_pos}
        Negative (non-bicycle) images: {len(images) - n_pos}
        """)

    def _ensure_rgb(self, image: Any) -> np.ndarray:
        """Convert image to RGB format."""
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                return np.stack([image] * 3, axis=-1)
            elif len(image.shape) == 3:
                if image.shape[-1] == 4:
                    return image[..., :3]
                elif image.shape[-1] == 3:
                    return image
                elif image.shape[-1] == 1:
                    return np.repeat(image, 3, axis=-1)
        elif isinstance(image, Image.Image):
            return np.array(image.convert("RGB"))

        raise ValueError(f"Unsupported image format: {type(image)}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[dict, int]:
        image = self.images[idx]
        label = self.labels[idx]

        # Ensure RGB format
        image = self._ensure_rgb(image)

        # Apply custom transform if provided
        if self.transform_fn is not None:
            image = self.transform_fn(image)

        # Process image for model
        processed = self.processor(images=image, return_tensors="pt")
        processed = {k: v.squeeze(0) for k, v in processed.items()}

        return processed, label


def create_dataloaders(
    processor: Any,
    config: Optional[DatasetConfig] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    config = config or DatasetConfig()

    train_dataset = BicycleDataset(processor, split="train", config=config)
    val_dataset = BicycleDataset(processor, split="val", config=config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    return train_loader, val_loader
