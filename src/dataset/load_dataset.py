from datasets import load_dataset, Dataset
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset as TorchDataset
import logging
from tqdm.auto import tqdm
from typing import Tuple, List, Optional
from PIL import Image

logger = logging.getLogger(__name__)

BICYCLE_LABEL = 2  # From the ClassLabel mapping


class BicycleDataset(TorchDataset):
    def __init__(self, images: List[Image.Image], labels: List[int]):
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        return self.images[idx], self.labels[idx]


def load_coco_dataset(split: str = "train") -> Dataset:
    """Load the COCO dataset from HuggingFace."""
    logger.info("Loading COCO dataset...")
    return load_dataset("rafaelpadilla/coco2017", split=split)


def filter_bicycle_images(dataset: Dataset) -> Tuple[Dataset, Dataset]:
    """Split dataset into bicycle and non-bicycle images."""
    logger.info("Filtering bicycle and non-bicycle images...")

    def contains_bicycle(example):
        return BICYCLE_LABEL in example["objects"]["label"]

    bicycle_dataset = dataset.filter(contains_bicycle, num_proc=4)
    non_bicycle_dataset = dataset.filter(lambda x: not contains_bicycle(x), num_proc=4)

    logger.info(
        f"Found {len(bicycle_dataset)} bicycle images and {len(non_bicycle_dataset)} non-bicycle images"
    )
    return bicycle_dataset, non_bicycle_dataset


def sample_balanced_indices(
    pos_size: int,
    neg_size: int,
    n_pos: int,
    n_neg: int,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample balanced indices for positive and negative examples."""
    if random_seed is not None:
        np.random.seed(random_seed)

    pos_indices = np.random.choice(pos_size, n_pos, replace=False)
    neg_indices = np.random.choice(neg_size, n_neg, replace=False)

    return pos_indices, neg_indices


def load_images(dataset: Dataset, indices: np.ndarray, desc: str) -> List[Image.Image]:
    """Load images from dataset given indices."""
    images = []
    # Convert numpy indices to regular Python integers
    indices = [int(idx) for idx in indices]
    for idx in tqdm(indices, desc=desc):
        img = dataset[idx]["image"]
        images.append(img)
    return images


def create_train_val_split(
    images: List[Image.Image],
    labels: List[int],
    train_ratio: float = 0.8,
    random_seed: Optional[int] = None,
) -> Tuple[BicycleDataset, BicycleDataset]:
    """Create training and validation datasets."""
    if random_seed is not None:
        np.random.seed(random_seed)

    # Shuffle data
    combined = list(zip(images, labels))
    np.random.shuffle(combined)
    images, labels = zip(*combined)

    # Split into train/val
    split_idx = int(len(images) * train_ratio)

    train_images = list(images[:split_idx])
    train_labels = list(labels[:split_idx])
    val_images = list(images[split_idx:])
    val_labels = list(labels[split_idx:])

    return (
        BicycleDataset(train_images, train_labels),
        BicycleDataset(val_images, val_labels),
    )


def load_or_create_cache(
    cache_file: Path,
) -> Optional[Tuple[BicycleDataset, BicycleDataset]]:
    """Load dataset from cache if it exists."""
    if cache_file.exists():
        logger.info("Loading cached dataset...")
        return torch.load(cache_file)
    return None


def save_to_cache(
    datasets: Tuple[BicycleDataset, BicycleDataset], cache_file: Path
) -> None:
    """Save datasets to cache."""
    logger.info("Caching datasets...")
    torch.save(datasets, cache_file)


def log_dataset_stats(
    train_dataset: BicycleDataset, val_dataset: BicycleDataset
) -> None:
    """Log statistics about the datasets."""
    total_images = len(train_dataset) + len(val_dataset)
    train_pos = sum(train_dataset.labels)
    val_pos = sum(val_dataset.labels)

    logger.info(f"""Dataset created:
    Total images: {total_images}
    Training images: {len(train_dataset)} (Positive: {train_pos}, Negative: {len(train_dataset)-train_pos})
    Validation images: {len(val_dataset)} (Positive: {val_pos}, Negative: {len(val_dataset)-val_pos})
    """)


def get_balanced_sample_sizes(
    pos_size: int, neg_size: int, max_images: int, neg_ratio: float
) -> Tuple[int, int]:
    """
    Calculate balanced sample sizes for positive and negative examples.

    Args:
        pos_size: Number of available positive examples
        neg_size: Number of available negative examples
        max_images: Maximum total number of images desired
        neg_ratio: Ratio of negative to positive examples

    Returns:
        Tuple of (n_positive, n_negative) counts to sample
    """
    # Calculate numbers ensuring integer division
    n_pos = min(pos_size, max_images // (1 + int(neg_ratio)))
    n_neg = int(n_pos * neg_ratio)

    # Ensure we don't try to sample more than available
    n_pos = min(n_pos, pos_size)
    n_neg = min(n_neg, neg_size)

    logger.info(f"Will sample {n_pos} positive and {n_neg} negative examples")
    return n_pos, n_neg


def get_bicycle_dataset(
    cache_dir: str = ".cache/bicycle_data",
    neg_ratio: float = 1.0,
    max_images: int = 1000,
    random_seed: Optional[int] = 42,
) -> Tuple[BicycleDataset, BicycleDataset]:
    """
    Download and prepare a balanced dataset of bicycle/non-bicycle images.

    Args:
        cache_dir: Directory to store the dataset
        neg_ratio: Ratio of negative to positive examples
        max_images: Maximum total number of images to download
        random_seed: Random seed for reproducibility

    Returns:
        tuple: (train_dataset, val_dataset) as BicycleDataset objects
    """
    # Input validation
    assert max_images > 0, "max_images must be positive"
    assert neg_ratio > 0, "neg_ratio must be positive"
    assert isinstance(max_images, int), "max_images must be an integer"
    # Setup cache
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"bicycle_data_{max_images}_{neg_ratio}.pt"

    # Try loading from cache
    cached_data = load_or_create_cache(cache_file)
    if cached_data is not None:
        return cached_data

    # Load and filter dataset
    dataset = load_coco_dataset()
    bicycle_dataset, non_bicycle_dataset = filter_bicycle_images(dataset)

    # Calculate sample sizes
    n_pos, n_neg = get_balanced_sample_sizes(
        pos_size=len(bicycle_dataset),
        neg_size=len(non_bicycle_dataset),
        max_images=max_images,
        neg_ratio=neg_ratio,
    )

    # Sample and load images
    pos_indices, neg_indices = sample_balanced_indices(
        len(bicycle_dataset), len(non_bicycle_dataset), n_pos, n_neg, random_seed
    )

    pos_images = load_images(bicycle_dataset, pos_indices, "Loading bicycle images")
    neg_images = load_images(
        non_bicycle_dataset, neg_indices, "Loading non-bicycle images"
    )

    # Create datasets
    all_images = pos_images + neg_images
    all_labels = [1] * len(pos_images) + [0] * len(neg_images)

    train_dataset, val_dataset = create_train_val_split(
        all_images, all_labels, train_ratio=0.8, random_seed=random_seed
    )

    # Cache and log
    save_to_cache((train_dataset, val_dataset), cache_file)
    log_dataset_stats(train_dataset, val_dataset)

    return train_dataset, val_dataset


if __name__ == "__main__":
    train_dataset, val_dataset = get_bicycle_dataset(
        max_images=3000, neg_ratio=1.0, random_seed=42
    )