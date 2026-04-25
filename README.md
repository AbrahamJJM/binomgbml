# BinomGBML

This repository contains the code to reproduce the main results from the paper "Binomial Gradient-Based Meta-Learning for Enhanced Meta-Gradient Estimation", presented in ICLR 2026.

[BinomGBML on arXiv](https://arxiv.org/abs/2604.13263) [BinomGBML OpenReview](https://openreview.net/forum?id=mKgUAO41zf)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/AbrahamJJM/binomgbml
cd binomgbml
```

2. Create Conda environment and install dependencies:

```bash
conda env create -f dependencies.yml
conda activate binomgbml
```

### Algorithm evaluation

Use the `main.py` script to train a model on miniImageNet or tieredImageNet using a meta-learning algorithm, then test it.

### Available Arguments

- `--dataset`: Dataset to use (mini-imagenet, tiered-imagenet)
- `--algorithm`: Algorithm to use (MAML, FOMAML, TruncMAML, BinomMAML, iMAML, Reptile)
- `--num-cls`: Number of classes per task (N-way)
- `--num-trn-data`: Number of training samples per class (K-shot)
- `--meta-lr`: Meta-learning rate
- `--task-lr`: Task-specific learning rate
- `--trunc`: Truncation parameter (for TruncMAML and BinomMAML)
- `--cuda`: Use CUDA if available
- `--seed`: Random seed

### Running Experiments

Execute the experiment script:

```bash
./experiments.sh
```

## Project Structure

```
.
├── main.py                 # Main training script
├── setup.sh               # Setup script for dependencies
├── experiments.sh         # Experiment configurations
├── pyproject.toml         # Python project configuration
├── src/
│   ├── meta_alg_base.py   # Base class for meta-learning algorithms
│   ├── utils.py           # Utility functions
│   └── algorithms/        # Algorithm implementations
│       ├── maml.py
│       ├── truncated_maml.py
│       ├── binomial_maml.py
│       ├── implicit_maml.py
│       └── reptile.py
└── README.md
```

## Citation
If you find this work useful, please consider citing:
> Y. Zhang, A. Jaeger Mountain, B. Li, and G. B. Giannakis, "Binomial Gradient-Based Meta-Learning for Enhanced Meta-Gradient Estimation,” in *Proceedings of the International Conference on Learning Representations (ICLR)*, 2026. 

```tex
@inproceedings{BinomGBML, 
  title={Binomial Gradient-Based Meta-Learning for Enhanced Meta-Gradient Estimation},
  author={Yilang Zhang and Abraham Jaeger Mountain and Bingcong Li and Georgios B. Giannakis},
  booktitle={The Fourteenth Annual International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=mKgUAO41zf}
}
