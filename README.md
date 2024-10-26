# HackOS-1

Our bike detection project for the HackOS-1 hackathon.

Dataset: [Coco minitrain](https://github.com/giddyyupp/coco-minitrain) (subset of [Coco](https://cocodataset.org/#home))

## Installation

This repository uses `uv`. Install [here](https://docs.astral.sh/uv/getting-started/installation/).

Install the repo with `git clone https://github.com/henri123lemoine/HackOS-bench.git`.

## Usage

```bash
uv run main.py
```

## TODO

- [x] Implement dataset caching
- [ ] Split into train and validation datasets (64/16 split?)
- [ ] Dataset augmentation
  - [ ] Random crop
  - [ ] Random flip
  - [ ] Random rotation
  - [ ] Random brightness
  - [ ] Random contrast
  - [ ] Random saturation
  - [ ] Random hue
- [ ] Test out various techniques and architectures:
  - [ ] SIFT with linear classifier
  - [ ] CNN
  - [ ] (Add ideas here)
- [ ] Random Forest
- [ ] Test benchmark

## Contributors

- [Amine Kobeissi](https://github.com/AKobeissi)
- [Bilguun Tegshbayar](https://github.com/Bilguun04)
- [Henri Lemoine](https://github.com/henri123lemoine)
