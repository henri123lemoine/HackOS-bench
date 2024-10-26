# Note from Laurence: I removed a couple of entires after checking a few for BICYCLE presence/absence.

import hashlib
from io import BytesIO
from pathlib import Path

import imageio.v3 as iio
import requests
import torch
import torch.nn.functional as F
import tqdm
from sklearn.metrics import accuracy_score

from src.dataset.url_list import LIST_OF_BICYCLES, LIST_OF_NON_BICYCLES


class ImageCache:
    def __init__(self):
        self.cache_dir = Path.home() / ".cache" / "betamark"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_image(self, url):
        # Create filename from URL hash
        url_hash = hashlib.md5(url.encode()).hexdigest()
        cache_path = self.cache_dir / f"{url_hash}.png"

        # Try loading from cache
        if cache_path.exists():
            try:
                return iio.imread(cache_path)
            except Exception as e:
                print(f"Error loading cached image: {e}")
                cache_path.unlink(missing_ok=True)

        # Download if not in cache
        try:
            res = requests.get(url)
            res.raise_for_status()
            img = iio.imread(BytesIO(res.content))
            iio.imwrite(cache_path, img, plugin="pillow")
            return img
        except Exception as e:
            print(f"Error downloading image: {e}")
            return None


def run_eval(user_func) -> dict:
    cache = ImageCache()
    correct_answers = 0
    total_answers = 0

    for url in tqdm.trange(len(LIST_OF_BICYCLES)):
        img = cache.get_image(LIST_OF_BICYCLES[url])
        total_answers += 1
        if user_func(img) == 1:
            correct_answers += 1

    for url in tqdm.trange(len(LIST_OF_NON_BICYCLES)):
        img = cache.get_image(LIST_OF_NON_BICYCLES[url])
        total_answers += 1
        if user_func(img) == 0:
            correct_answers += 1

    return {"acc": correct_answers / total_answers}


def evaluate_model(model, dataloader, device):
    model.eval()  # Set model to evaluation mode
    all_labels = []
    all_predictions = []

    with torch.no_grad():  # No need to calculate gradients
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            predictions = probabilities.argmax(dim=1).cpu().numpy()

            # Collect labels and predictions
            all_labels.extend(labels.numpy())
            all_predictions.extend(predictions)

    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    return accuracy
