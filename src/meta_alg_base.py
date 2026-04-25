import os
from argparse import Namespace
from abc import ABC, abstractmethod
from typing import Sequence, Optional

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.func as func
from learn2learn.data import Taskset, partition_task
from learn2learn.vision.benchmarks import get_tasksets
from learn2learn.vision.models import CNN4, ResNet12, WRN28

from src.utils import Checkpointer


_BASE_MODELS = {
    "cnn4": CNN4, 
    "resnet12": ResNet12, 
    "wrn28": WRN28
    }


class MetaLearningAlgBase(ABC):
    @abstractmethod
    def __init__(self, args: Namespace) -> None:
        self.args = args

        self.meta_trn_dataset, self.meta_val_dataset, self.meta_tst_dataset = self._get_meta_datasets()
        self.base_model = self._get_base_model(device="meta")
        self.meta_model = self._get_meta_model(device=self.args.device)
        self.nll = nn.CrossEntropyLoss()

    def _get_meta_datasets(self) -> tuple[Taskset, Taskset, Taskset]:
        return get_tasksets(
            self.args.dataset.lower(),
            train_ways=self.args.num_cls,
            train_samples=self.args.num_trn_data + self.args.num_val_data,
            test_ways=self.args.num_cls,
            test_samples=self.args.num_trn_data * 2,
            root=self.args.data_dir,
        )

    def sample_task_data(
            self, meta_dataset: Taskset
            ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        (trn_inputs, trn_targets), (val_inputs, val_targets) = partition_task(
            *meta_dataset.sample(), shots=self.args.num_trn_data
        )
        device = self.args.device
        trn_inputs, trn_targets = trn_inputs.to(device), trn_targets.to(device)
        val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)

        return (trn_inputs, trn_targets), (val_inputs, val_targets)

    def _get_base_model(self, device: Optional[str | torch.device] = None, **kwargs) -> nn.Module:
        if self.args.base_model.lower() == "cnn4":
            kwargs["hidden_size"] = self.args.hidden_size
        model = _BASE_MODELS[self.args.base_model.lower()](output_size=self.args.num_cls, **kwargs).to(device)
        func.replace_all_batch_norm_modules_(model)  # transductive

        return model

    @abstractmethod
    def _get_meta_model(self, **kwargs) -> nn.Module:
        raise NotImplementedError

    def _get_meta_optimizer(self) -> optim.Optimizer:
        return torch.optim.Adam(self.meta_model.parameters(), self.args.meta_lr)

    @abstractmethod
    def adapt(self, trn_inputs: torch.Tensor, trn_targets: torch.Tensor, 
              first_order: bool = False) -> dict[str, any]:
        raise NotImplementedError

    def save_meta_model(self, file_name: str) -> None:
        torch.save(self.meta_model.state_dict(), os.path.join(self.args.model_dir, file_name))

    def load_meta_model(self, file_name: str) -> None:
        self.meta_model.load_state_dict(torch.load(os.path.join(self.args.model_dir, file_name), weights_only=True))

    def train(self) -> None:
        print("Training starts ...")
        meta_optimizer = self._get_meta_optimizer()
        checkpointer = Checkpointer(self.save_meta_model, self.args.algorithm.lower())

        running_loss = 0.0
        running_acc = 0.0
        self.base_model.train()

        # training loop
        for meta_idx in range(self.args.meta_iter):
            self.meta_model.train()
            meta_optimizer.zero_grad()

            for _ in range(self.args.meta_batch_size):
                (trn_inputs, trn_targets), (val_inputs, val_targets) = self.sample_task_data(self.meta_trn_dataset)
                adapted_kwargs = self.adapt(trn_inputs, trn_targets, first_order=self.args.first_order)
                val_logits = func.functional_call(self.base_model, adapted_kwargs['named_params'], val_inputs)
                meta_loss = self.nll(val_logits, val_targets)
                self.meta_backward(meta_loss, adapted_kwargs)

                with torch.no_grad():
                    running_loss += meta_loss.item()
                    running_acc += (val_logits.argmax(dim=1) == val_targets).float().mean().item()

            meta_optimizer.step()

            # meta-validation
            if (meta_idx + 1) % self.args.log_iter == 0:
                print(
                    f"Meta-iter {meta_idx + 1}: "
                    f"train loss = {running_loss / self.args.log_iter  / self.args.meta_batch_size:.3f}, "
                    f"train acc = {running_acc / (self.args.log_iter * self.args.meta_batch_size) * 100:.2f}%, "
                )
                running_loss, running_acc = 0., 0.

            # save
            if (meta_idx + 1) % self.args.save_iter == 0:
                val_loss, val_acc = self.evaluate(self.meta_val_dataset, self.args.num_val_tasks)
                checkpointer.update(val_acc)
                print(
                    f"Checkpoint {checkpointer.counter}: val loss = {val_loss:.4f}, " f"val acc = {val_acc * 100:.2f}%"
                )

    def test(self) -> None:
        print("Testing starts ...")
        loss_mean, loss_95ci, acc_mean, acc_95ci = self.evaluate(
            self.meta_tst_dataset, self.args.num_tst_tasks, return_ci=True
        )
        print(
            f"Test loss = {loss_mean:.4f} +/- {loss_95ci:.4f}, "
            f"test acc = {acc_mean * 100:.2f}% +/- {acc_95ci * 100:.2f}%"
        )

    def evaluate(self, meta_dataset: Taskset, num_tasks: int, return_ci: bool = False) -> Sequence[float]:
        self.meta_model.eval()
        loss_list = list()
        acc_list = list()

        for _ in range(num_tasks):
            (trn_inputs, trn_targets), (tst_inputs, tst_targets) = self.sample_task_data(meta_dataset)
            adapted_kwargs = self.adapt(trn_inputs, trn_targets, first_order=True)
            with torch.no_grad():
                tst_logits = func.functional_call(self.base_model, adapted_kwargs['named_params'], tst_inputs)
                loss_list.append(self.nll(tst_logits, tst_targets).item())
                acc_list.append((tst_logits.argmax(dim=1) == tst_targets).float().mean().item())

        if return_ci:
            return (
                np.mean(loss_list),
                np.std(loss_list) / np.sqrt(num_tasks),
                np.mean(acc_list),
                np.std(acc_list) / np.sqrt(num_tasks),
            )
        else:
            return np.mean(loss_list), np.mean(acc_list)

    def meta_backward(self, meta_loss: torch.Tensor, adapted_kwargs: Optional[dict[str, any]] = None) -> None:
        meta_loss.backward()

    def is_meta_training(self) -> bool:
        return self.meta_model.training
