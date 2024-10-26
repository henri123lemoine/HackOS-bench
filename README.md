# HackOS-1

Our bike detection project for the HackOS-1 hackathon.

Dataset: [Coco minitrain](https://github.com/giddyyupp/coco-minitrain) (subset of [Coco](https://cocodataset.org))

## Installation

This repository uses `uv`. Install [here](https://docs.astral.sh/uv/getting-started/installation/).

Install the repo with `git clone https://github.com/henri123lemoine/HackOS-bench.git`.

## Usage

To run `main.py`:
```bash
uv run main.py
```

## TODO

### Dataset

- [x] [Coco](https://cocodataset.org) (subset -> [Coco minitrain](https://github.com/giddyyupp/coco-minitrain))
- [x] Downloading
- [ ] Processing
- [ ] Split into train and validation datasets
- [ ] Dataset augmentation (Unclear if worthwhile with large enough dataset)

### Models

- [ ] Test out general-purpose models:
  - [ ] ViT?
  - [ ] ?
- [ ] Finetuning pipeline
  - [ ] Look into sample-efficient finetuning
  - [ ] Implement ^^
- [ ] Random Forest for combining various models(?)

## Contributors

- [Amine Kobeissi](https://github.com/AKobeissi)
- [Bilguun Tegshbayar](https://github.com/Bilguun04)
- [Henri Lemoine](https://github.com/henri123lemoine)
- [Danil Garmaev](https://github.com/danilgarmaev)
