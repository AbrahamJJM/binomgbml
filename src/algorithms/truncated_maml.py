from collections import OrderedDict
import torch
import torch.nn as nn
from torch.func import functional_call
from src.meta_alg_base import MetaLearningAlgBase


class TruncMAML(MetaLearningAlgBase):
    def __init__(self, args) -> None:
        super().__init__(args)
        assert 0 <= self.args.trunc <= self.args.task_iter_trn

    def _get_meta_model(self, **kwargs) -> nn.Module:
        return self._get_base_model(**kwargs)

    def adapt(self, trn_inputs: torch.Tensor, trn_targets: torch.Tensor, 
              first_order: bool = False) -> dict[str, any]:
        named_params = OrderedDict(self.meta_model.named_parameters())
        meta_training = self.is_meta_training()
        task_iter = self.args.task_iter_trn if meta_training else self.args.task_iter_eval

        for trn_idx in range(task_iter):
            trn_logits = functional_call(self.base_model, 
                                         named_params, 
                                         trn_inputs)
            trn_loss = self.nll(trn_logits, trn_targets)
            trn_grads = torch.autograd.grad(
                trn_loss, named_params.values(), 
                create_graph=(not first_order) and meta_training and (trn_idx >= task_iter - self.args.trunc)
            )
            for (name, param), grad in zip(named_params.items(), trn_grads):
                named_params[name] = (param - self.args.task_lr * grad).requires_grad_()

        return {'named_params': named_params}
