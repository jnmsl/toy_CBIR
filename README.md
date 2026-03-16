# Fashion Product CBIR

Content-Based Image Retrieval for fashion e-commerce. Upload a product photo, get visually similar items.

## Descriptors

| Block | Method | Dims | Purpose |
|-------|--------|------|---------|
| Semantic | CNN (ResNet50 / ImageNet, avg pooling) | 2048 | Semantic embedding |
| Color | HSV Color Histogram (32 bins/channel) | 96 | Color distribution |
| Texture | LBP (P=24, R=3, uniform) | 26 | Texture / fabric |

Each block is L2-normalized independently.

## Search Strategies

Default strategy is **cascaded** two-stage retrieval:

1. **Stage 1** — Cosine distance on CNN embeddings selects top-100 candidates (`NearestNeighbors`, brute-force).
2. **Stage 2** — Re-ranks candidates by `d_color + 0.5 * d_texture` (Euclidean).

Two single-feature baselines are also available via `--strategy`:

| Strategy | Metric | Features |
|----------|--------|----------|
| `cascaded` (default) | Cosine + Euclidean | All three |
| `resnet_only` | Cosine | CNN only |
| `color_only` | Euclidean | HSV histogram only |

## Dataset

[Kaggle Fashion Product Images](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) (~44k products). Metadata in `archive/styles.csv`.

## Usage

```bash
# Full pipeline (index + evaluate + demo)
python ToyCBIR.py --mode all --image-dir ./images --metadata ./archive/styles.csv

# Single query
python ToyCBIR.py --mode search --query ./images/42431.jpg --top-k 10

# Evaluate a specific strategy
python ToyCBIR.py --mode evaluate --strategy resnet_only

# On CeSViMa Magerit
sbatch submit_job.sh
```

## Requirements

Python 3.9+. Install in conda env:

```bash
pip install tensorflow opencv-python opencv-contrib-python scikit-image scikit-learn matplotlib tqdm scipy
```
