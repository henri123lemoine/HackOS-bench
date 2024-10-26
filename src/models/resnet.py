import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import requests
import cv2
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os
import hashlib
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Test images
TEST_IMAGES = {
    "bicycle": [
        "http://images.cocodataset.org/train2017/000000483108.jpg",
        "http://images.cocodataset.org/train2017/000000293802.jpg",
        "http://images.cocodataset.org/train2017/000000079841.jpg",
        "http://images.cocodataset.org/train2017/000000515289.jpg",
        "http://images.cocodataset.org/train2017/000000562150.jpg",
        "http://images.cocodataset.org/train2017/000000412151.jpg",
        "http://images.cocodataset.org/train2017/000000462565.jpg",
        "http://images.cocodataset.org/train2017/000000509822.jpg",
        "http://images.cocodataset.org/train2017/000000321107.jpg",
        "http://images.cocodataset.org/train2017/000000061181.jpg",
        "http://images.cocodataset.org/train2017/000000018783.jpg",
        "http://images.cocodataset.org/train2017/000000012896.jpg",
        "http://images.cocodataset.org/train2017/000000465692.jpg",
        "http://images.cocodataset.org/train2017/000000391584.jpg",
        "http://images.cocodataset.org/train2017/000000241350.jpg",
        "http://images.cocodataset.org/train2017/000000438024.jpg",
        "http://images.cocodataset.org/train2017/000000442726.jpg",
        "http://images.cocodataset.org/train2017/000000435937.jpg",
        "http://images.cocodataset.org/train2017/000000292819.jpg",
        "http://images.cocodataset.org/train2017/000000157416.jpg",
        "http://images.cocodataset.org/train2017/000000010393.jpg",
        "http://images.cocodataset.org/train2017/000000084540.jpg",
        "http://images.cocodataset.org/train2017/000000007125.jpg",
        "http://images.cocodataset.org/train2017/000000507249.jpg",
        "http://images.cocodataset.org/train2017/000000075923.jpg",
        "http://images.cocodataset.org/train2017/000000240918.jpg",
        "http://images.cocodataset.org/train2017/000000122302.jpg",
        "http://images.cocodataset.org/train2017/000000140006.jpg",
        "http://images.cocodataset.org/train2017/000000536444.jpg",
        "http://images.cocodataset.org/train2017/000000344271.jpg",
        "http://images.cocodataset.org/train2017/000000420081.jpg",
        "http://images.cocodataset.org/train2017/000000148668.jpg",
        "http://images.cocodataset.org/train2017/000000390137.jpg",
        "http://images.cocodataset.org/train2017/000000114183.jpg",
        "http://images.cocodataset.org/train2017/000000020307.jpg",
        "http://images.cocodataset.org/train2017/000000280736.jpg",
        "http://images.cocodataset.org/train2017/000000536321.jpg",
        "http://images.cocodataset.org/train2017/000000188146.jpg",
        "http://images.cocodataset.org/train2017/000000559312.jpg",
        "http://images.cocodataset.org/train2017/000000535808.jpg",
        "http://images.cocodataset.org/train2017/000000451944.jpg",
        "http://images.cocodataset.org/train2017/000000212558.jpg",
        "http://images.cocodataset.org/train2017/000000377867.jpg",
        "http://images.cocodataset.org/train2017/000000139291.jpg",
        "http://images.cocodataset.org/train2017/000000456323.jpg",
        "http://images.cocodataset.org/train2017/000000549386.jpg",
        "http://images.cocodataset.org/train2017/000000254491.jpg",
        "http://images.cocodataset.org/train2017/000000314515.jpg",
        "http://images.cocodataset.org/train2017/000000415904.jpg",
        "http://images.cocodataset.org/train2017/000000101636.jpg",
        "http://images.cocodataset.org/train2017/000000315173.jpg",
        "http://images.cocodataset.org/train2017/000000260627.jpg",
        "http://images.cocodataset.org/train2017/000000001722.jpg",
        "http://images.cocodataset.org/train2017/000000031092.jpg",
        "http://images.cocodataset.org/train2017/000000556205.jpg",
        "http://images.cocodataset.org/train2017/000000049097.jpg",
        "http://images.cocodataset.org/train2017/000000070815.jpg",
        "http://images.cocodataset.org/train2017/000000467000.jpg",
        "http://images.cocodataset.org/train2017/000000416733.jpg",
        "http://images.cocodataset.org/train2017/000000203912.jpg",
        "http://images.cocodataset.org/train2017/000000408143.jpg",
        "http://images.cocodataset.org/train2017/000000120340.jpg",
        "http://images.cocodataset.org/train2017/000000124462.jpg",
        "http://images.cocodataset.org/train2017/000000142718.jpg",
        "http://images.cocodataset.org/train2017/000000108838.jpg",
        "http://images.cocodataset.org/train2017/000000445309.jpg",
        "http://images.cocodataset.org/train2017/000000140197.jpg",
        "http://images.cocodataset.org/train2017/000000012993.jpg",
        "http://images.cocodataset.org/train2017/000000111099.jpg",
        "http://images.cocodataset.org/train2017/000000215867.jpg",
        "http://images.cocodataset.org/train2017/000000565085.jpg",
        "http://images.cocodataset.org/train2017/000000314986.jpg",
        "http://images.cocodataset.org/train2017/000000158708.jpg",
        "http://images.cocodataset.org/train2017/000000263961.jpg",
        "http://images.cocodataset.org/train2017/000000192128.jpg",
        "http://images.cocodataset.org/train2017/000000377832.jpg",
        "http://images.cocodataset.org/train2017/000000187286.jpg",
        "http://images.cocodataset.org/train2017/000000195510.jpg",
        "http://images.cocodataset.org/train2017/000000406949.jpg",
        "http://images.cocodataset.org/train2017/000000330455.jpg",
    ],
    "non_bicycle": [
        "http://images.cocodataset.org/train2017/000000522418.jpg",
        "http://images.cocodataset.org/train2017/000000184613.jpg",
        "http://images.cocodataset.org/train2017/000000318219.jpg",
        "http://images.cocodataset.org/train2017/000000554625.jpg",
        "http://images.cocodataset.org/train2017/000000574769.jpg",
        "http://images.cocodataset.org/train2017/000000060623.jpg",
        "http://images.cocodataset.org/train2017/000000309022.jpg",
        "http://images.cocodataset.org/train2017/000000005802.jpg",
        "http://images.cocodataset.org/train2017/000000222564.jpg",
        "http://images.cocodataset.org/train2017/000000118113.jpg",
        "http://images.cocodataset.org/train2017/000000193271.jpg",
        "http://images.cocodataset.org/train2017/000000224736.jpg",
        "http://images.cocodataset.org/train2017/000000403013.jpg",
        "http://images.cocodataset.org/train2017/000000374628.jpg",
        "http://images.cocodataset.org/train2017/000000328757.jpg",
        "http://images.cocodataset.org/train2017/000000384213.jpg",
        "http://images.cocodataset.org/train2017/000000086408.jpg",
        "http://images.cocodataset.org/train2017/000000372938.jpg",
        "http://images.cocodataset.org/train2017/000000386164.jpg",
        "http://images.cocodataset.org/train2017/000000223648.jpg",
        "http://images.cocodataset.org/train2017/000000204805.jpg",
        "http://images.cocodataset.org/train2017/000000113588.jpg",
        "http://images.cocodataset.org/train2017/000000384553.jpg",
        "http://images.cocodataset.org/train2017/000000337264.jpg",
        "http://images.cocodataset.org/train2017/000000368402.jpg",
        "http://images.cocodataset.org/train2017/000000012448.jpg",
        "http://images.cocodataset.org/train2017/000000542145.jpg",
        "http://images.cocodataset.org/train2017/000000540186.jpg",
        "http://images.cocodataset.org/train2017/000000242611.jpg",
        "http://images.cocodataset.org/train2017/000000051191.jpg",
        "http://images.cocodataset.org/train2017/000000269105.jpg",
        "http://images.cocodataset.org/train2017/000000294832.jpg",
        "http://images.cocodataset.org/train2017/000000144941.jpg",
        "http://images.cocodataset.org/train2017/000000173350.jpg",
        "http://images.cocodataset.org/train2017/000000060760.jpg",
        "http://images.cocodataset.org/train2017/000000324266.jpg",
        "http://images.cocodataset.org/train2017/000000166532.jpg",
        "http://images.cocodataset.org/train2017/000000262284.jpg",
        "http://images.cocodataset.org/train2017/000000360772.jpg",
        "http://images.cocodataset.org/train2017/000000191381.jpg",
        "http://images.cocodataset.org/train2017/000000111076.jpg",
        "http://images.cocodataset.org/train2017/000000340559.jpg",
        "http://images.cocodataset.org/train2017/000000258985.jpg",
        "http://images.cocodataset.org/train2017/000000229643.jpg",
        "http://images.cocodataset.org/train2017/000000125059.jpg",
        "http://images.cocodataset.org/train2017/000000455483.jpg",
        "http://images.cocodataset.org/train2017/000000436141.jpg",
        "http://images.cocodataset.org/train2017/000000129001.jpg",
        "http://images.cocodataset.org/train2017/000000232262.jpg",
        "http://images.cocodataset.org/train2017/000000166323.jpg",
        "http://images.cocodataset.org/train2017/000000580041.jpg",
        "http://images.cocodataset.org/train2017/000000326781.jpg",
        "http://images.cocodataset.org/train2017/000000387362.jpg",
        "http://images.cocodataset.org/train2017/000000138079.jpg",
        "http://images.cocodataset.org/train2017/000000556616.jpg",
        "http://images.cocodataset.org/train2017/000000472621.jpg",
        "http://images.cocodataset.org/train2017/000000192440.jpg",
        "http://images.cocodataset.org/train2017/000000086320.jpg",
        "http://images.cocodataset.org/train2017/000000256668.jpg",
        "http://images.cocodataset.org/train2017/000000383445.jpg",
        "http://images.cocodataset.org/train2017/000000565797.jpg",
        "http://images.cocodataset.org/train2017/000000081922.jpg",
        "http://images.cocodataset.org/train2017/000000050125.jpg",
        "http://images.cocodataset.org/train2017/000000364521.jpg",
        "http://images.cocodataset.org/train2017/000000394892.jpg",
        "http://images.cocodataset.org/train2017/000000001146.jpg",
        "http://images.cocodataset.org/train2017/000000310391.jpg",
        "http://images.cocodataset.org/train2017/000000097434.jpg",
        "http://images.cocodataset.org/train2017/000000463836.jpg",
        "http://images.cocodataset.org/train2017/000000241876.jpg",
        "http://images.cocodataset.org/train2017/000000156832.jpg",
        "http://images.cocodataset.org/train2017/000000270721.jpg",
        "http://images.cocodataset.org/train2017/000000462341.jpg",
        "http://images.cocodataset.org/train2017/000000310103.jpg",
        "http://images.cocodataset.org/train2017/000000032992.jpg",
        "http://images.cocodataset.org/train2017/000000122851.jpg",
        "http://images.cocodataset.org/train2017/000000540763.jpg",
        "http://images.cocodataset.org/train2017/000000138246.jpg",
        "http://images.cocodataset.org/train2017/000000197254.jpg",
        "http://images.cocodataset.org/train2017/000000032907.jpg",
    ],
}


class BicycleDataset(Dataset):
    def __init__(
        self, image_urls, labels=None, transform=None, train=True, cache_dir=None
    ):
        self.image_urls = image_urls
        self.labels = (
            labels if labels is not None else [0] * len(image_urls)
        )  # Default to bicycle class if no labels
        self.train = train
        self.cache_dir = cache_dir

        if transform is None:
            self.transform = self.get_default_transforms(train)
        else:
            self.transform = transform

        # Create cache directory if it doesn't exist
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.image_urls)

    def _get_cached_image_path(self, url):
        """Generate a cached file path for a given URL"""
        if self.cache_dir is None:
            return None
        filename = hashlib.md5(url.encode()).hexdigest() + ".jpg"
        return os.path.join(self.cache_dir, filename)

    def _load_image(self, url):
        """Load image from URL or cache"""
        cache_path = self._get_cached_image_path(url)

        # Try to load from cache first
        if cache_path and os.path.exists(cache_path):
            return cv2.imread(cache_path)

        # If not in cache, download and save
        try:
            response = requests.get(url)
            image = np.asarray(bytearray(response.content), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)

            # Cache the image if caching is enabled
            if cache_path and image is not None:
                cv2.imwrite(cache_path, image)

            return image
        except Exception as e:
            print(f"Error loading image {url}: {str(e)}")
            return None

    def __getitem__(self, idx):
        # Load image
        image = self._load_image(self.image_urls[idx])
        if image is None:
            # Return a dummy image if loading fails
            image = np.zeros((224, 224, 3), dtype=np.uint8)

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, self.labels[idx]

    @staticmethod
    def get_default_transforms(train=True):
        if train:
            return A.Compose(
                [
                    A.RandomResizedCrop(224, 224, scale=(0.8, 1.0)),
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=30, p=0.5),
                    A.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3
                    ),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ]
            )
        else:
            return A.Compose(
                [
                    A.Resize(224, 224),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ]
            )


class ModelEvaluator:
    def __init__(self, model, device, class_names=["bicycle", "non_bicycle"]):
        self.model = model
        self.device = device
        self.class_names = class_names
        self.misclassified_examples = []

    def denormalize_image(self, image):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return image * std + mean

    def evaluate(self, dataloader):
        self.model.eval()
        all_preds = []
        all_labels = []
        all_confidences = []

        print("Evaluating model...")
        with torch.no_grad():
            for images, labels in tqdm(dataloader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                confidences = torch.max(probabilities, dim=1)[0]

                misclassified_mask = predictions != labels
                for idx in range(len(images)):
                    if misclassified_mask[idx]:
                        self.misclassified_examples.append(
                            {
                                "image": images[idx].cpu(),
                                "true_label": labels[idx].item(),
                                "predicted_label": predictions[idx].item(),
                                "confidence": confidences[idx].item(),
                            }
                        )

                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())

        return self.analyze_results(all_preds, all_labels, all_confidences)

    def analyze_results(self, predictions, true_labels, confidences):
        cm = confusion_matrix(true_labels, predictions)
        report = classification_report(
            true_labels, predictions, target_names=self.class_names, output_dict=True
        )

        correct_mask = np.array(predictions) == np.array(true_labels)
        correct_confidences = np.array(confidences)[correct_mask]
        incorrect_confidences = np.array(confidences)[~correct_mask]

        results = {
            "confusion_matrix": cm,
            "classification_report": report,
            "confidence_stats": {
                "correct_mean": np.mean(correct_confidences)
                if len(correct_confidences) > 0
                else 0,
                "correct_std": np.std(correct_confidences)
                if len(correct_confidences) > 0
                else 0,
                "incorrect_mean": np.mean(incorrect_confidences)
                if len(incorrect_confidences) > 0
                else 0,
                "incorrect_std": np.std(incorrect_confidences)
                if len(incorrect_confidences) > 0
                else 0,
            },
        }

        return results

    def plot_confusion_matrix(self, results):
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            results["confusion_matrix"],
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
        )
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.show()

    def plot_misclassified_examples(self, num_examples=10):
        n = min(num_examples, len(self.misclassified_examples))
        if n == 0:
            print("No misclassified examples found!")
            return

        fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(20, 8))
        if n == 1:
            axes = [axes]
        else:
            axes = axes.ravel()

        for idx, example in enumerate(self.misclassified_examples[:n]):
            img = self.denormalize_image(example["image"])
            img = img.permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)

            axes[idx].imshow(img)
            axes[idx].axis("off")
            axes[idx].set_title(
                f'True: {self.class_names[example["true_label"]]}\n'
                f'Pred: {self.class_names[example["predicted_label"]]}\n'
                f'Conf: {example["confidence"]:.2f}'
            )

        plt.tight_layout()
        plt.show()

    def print_performance_summary(self, results):
        print("\n=== Model Performance Summary ===")

        # Overall metrics
        print("\nOverall Metrics:")
        report = results["classification_report"]
        print(f"Accuracy: {report['accuracy']:.3f}")
        print(f"Macro Avg F1-Score: {report['macro avg']['f1-score']:.3f}")

        # Per-class performance
        print("\nPer-class Performance:")
        for class_name in self.class_names:
            metrics = report[class_name.lower()]
            print(f"\n{class_name}:")
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall: {metrics['recall']:.3f}")
            print(f"  F1-Score: {metrics['f1-score']:.3f}")

        # Confidence analysis
        conf_stats = results["confidence_stats"]
        print("\nConfidence Analysis:")
        print(
            f"Correct predictions - Mean: {conf_stats['correct_mean']:.3f}, Std: {conf_stats['correct_std']:.3f}"
        )
        print(
            f"Incorrect predictions - Mean: {conf_stats['incorrect_mean']:.3f}, Std: {conf_stats['incorrect_std']:.3f}"
        )


def create_test_dataset():
    images = []
    labels = []

    # Add bicycle images (label 0)
    for url in TEST_IMAGES["bicycle"]:
        images.append(url)
        labels.append(0)

    # Add motorcycle images (label 1)
    for url in TEST_IMAGES["non_bicycle"]:
        images.append(url)
        labels.append(1)

    test_dataset = BicycleDataset(
        image_urls=images,
        labels=labels,
        train=False,
        cache_dir=Path("data/.cache/bicycle_data"),
    )

    return test_dataset


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    print("Loading model...")
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(
        model.fc.in_features, 2
    )  # Binary classification efficientnet/mobile net
    model = model.to(device)

    # Create test dataset and loader
    print("Creating test dataset...")
    test_dataset = create_test_dataset()
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)

    # Create evaluator
    print("Starting evaluation...")
    evaluator = ModelEvaluator(model, device)

    # Run evaluation
    results = evaluator.evaluate(test_loader)

    # Display results
    evaluator.print_performance_summary(results)
    evaluator.plot_confusion_matrix(results)
    evaluator.plot_misclassified_examples()


if __name__ == "__main__":
    main()
