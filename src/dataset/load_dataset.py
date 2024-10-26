"""
from datasets import load_dataset
coco = load_dataset("detection-datasets/coco")
bicycle_data = coco.filter(lambda x: 'bicycle' in x['categories'])

imagenet = load_dataset("imagenet-1k")

from torchvision.datasets import CocoDetection, ImageNet
coco = CocoDetection(root='path/to/coco', annFile='path/to/annotations')
"""
