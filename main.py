import os
import random
import argparse

import numpy as np
import torch

from src.algorithms.maml import MAML, FOMAML
from src.algorithms.truncated_maml import TruncMAML
from src.algorithms.binomial_maml import BinomMAML
from src.algorithms.implicit_maml import iMAML
from src.algorithms.reptile import Reptile


_ALGORITHM = {
    "maml": MAML,
    "fomaml": FOMAML,
    "truncmaml": TruncMAML,
    "binommaml": BinomMAML,
    "imaml": iMAML,
    'reptile': Reptile,
    # You can add your own algorithm here
    }


def main(args: argparse.Namespace) -> None:
    args.data_dir = os.path.join(args.data_dir, args.dataset.lower())
    # os.makedirs(args.data_dir, exist_ok=True)

    suffix = "-" + str(args.num_cls) + "way" + str(args.num_trn_data) + "shot"
    args.model_dir = os.path.join(args.model_dir, args.dataset.lower(), args.algorithm.lower() + suffix)
    os.makedirs(args.model_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # for CUDA >= 10.2
    torch.use_deterministic_algorithms(True)

    for k, v in args.__dict__.items():
        print(f"{k}: {v}")

    args.device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    algorithm = _ALGORITHM[args.algorithm.lower()](args)
    algorithm.train()
    algorithm.load_meta_model(args.algorithm.lower() + "_final.ct")
    algorithm.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup variables")

    # dataset
    parser.add_argument("--dataset", type=str, default="mini-ImageNet", help="Dataset name")
    parser.add_argument("--data-dir", type=str, default="./datasets/", help="Dataset directory")
    parser.add_argument("--num-cls", type=int, default=5, help="Number of classes (way) per task")
    parser.add_argument("--num-trn-data", type=int, default=1, help="Number of training data per class")
    parser.add_argument("--num-val-data", type=int, default=15, help="Number of validation data per class")
    parser.add_argument("--num-val-tasks", type=int, default=600, help="Number of meta-validation tasks")
    parser.add_argument("--num-tst-tasks", type=int, default=600, help="Number of meta-testing tasks")
    parser.add_argument("--seed", type=int, default=0, help="Seed for reproducibility")
    parser.add_argument("--cuda", type=bool, default=True, help="Whether to use cuda")

    # meta-training params
    parser.add_argument("--model-dir", type=str, default="./models/", help="Save directory")
    parser.add_argument("--first-order", action="store_true", help="Whether to use first-order approximation")
    parser.add_argument("--meta-iter", type=int, default=20000, help="Number of iters for meta-training")
    parser.add_argument("--meta-batch-size", type=int, default=4, help="Batch size of tasks to update meta-param")
    parser.add_argument("--log-iter", type=int, default=100, help="Log iter")
    parser.add_argument("--save-iter", type=int, default=1000, help="Save iter")
    parser.add_argument("--meta-lr", type=float, default=1, help="Learning rate for meta-updates")

    # task-training params
    parser.add_argument("--algorithm", type=str, default="BinomMAML", help="Few-shot learning methods")
    parser.add_argument("--base-model", type=str, default="CNN4", help="Backbone model")
    parser.add_argument("--hidden-size", type=int, default=32, help="Number of filters per layer in CNN4")
    parser.add_argument("--task-iter-trn", type=int, default=5, help="Number of adaptation steps")
    parser.add_argument("--task-iter-eval", type=int, default=10, help="Number of adaptation steps")
    parser.add_argument("--task-lr", type=float, default=1e-2, help="Learning rate for adaptation")
    parser.add_argument("--trunc", type=int, default=2, help="Number of truncated steps for TruncMAML/BinMAML")
    parser.add_argument("--lambd", type=float, default=1e2, help="Regularization coefficient for iMAML")

    args_parsed = parser.parse_args()
    main(args_parsed)
