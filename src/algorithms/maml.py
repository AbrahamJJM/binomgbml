from collections import OrderedDict
import torch
import torch.nn as nn
from torch.func import functional_call
from src.meta_alg_base import MetaLearningAlgBase


class MAML(MetaLearningAlgBase):
    def __init__(self, args) -> None:
        super().__init__(args)

    def _get_meta_model(self, **kwargs) -> nn.Module:
        return self._get_base_model(**kwargs)

    def adapt(self, trn_inputs: torch.Tensor, trn_targets: torch.Tensor, 
              first_order: bool = False) -> dict[str, any]:
        named_params = OrderedDict(self.meta_model.named_parameters())

        task_iter = self.args.task_iter_trn if self.is_meta_training() else self.args.task_iter_eval
        for _ in range(task_iter):
            trn_logits = functional_call(self.base_model, 
                                         named_params, 
                                         trn_inputs)
            trn_loss = self.nll(trn_logits, trn_targets)
            trn_grads = torch.autograd.grad(trn_loss, 
                                            named_params.values(), 
                                            create_graph=not first_order)
            for (name, param), grad in zip(named_params.items(), trn_grads):
                named_params[name] = param - self.args.task_lr * grad

        return {'named_params': named_params}


class FOMAML(MAML):
    def __init__(self, args) -> None:
        args.first_order = True
        super().__init__(args)
