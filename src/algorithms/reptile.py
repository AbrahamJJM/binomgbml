from collections import OrderedDict
import torch
import torch.nn as nn
from torch.func import functional_call
from src.meta_alg_base import MetaLearningAlgBase

class Reptile_Meta_Optimizer(torch.optim.Optimizer):
    
    def __init__(self, params, lr=0.001):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    # Assumes grads are populated with the difference between adapted and original parameters
    def step(self):
        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    p.data.add_(p.grad, alpha=group['lr'])

    def zero_grad(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()

class Reptile(MetaLearningAlgBase):
    def __init__(self, args) -> None:
        super().__init__(args)

    def _get_meta_model(self, **kwargs) -> nn.Module:
        return self._get_base_model(**kwargs)
    
    # Have to override due to special meta-optimization of Reptile
    def _get_meta_optimizer(self) -> torch.optim.Optimizer:
        return Reptile_Meta_Optimizer(self.meta_model.parameters(), lr=self.args.meta_lr)
    
    @torch.no_grad()
    def meta_backward(self, meta_loss, adapted_kwargs = None):
        # Rather than directly backpropagating, we will update the model parameters' gradients directly
        adapted_parameters = adapted_kwargs['named_params']
        with torch.no_grad():
            for name, param in self.meta_model.named_parameters():
                if param.grad is None:
                    param.grad = (adapted_parameters[name] - param) / self.args.meta_batch_size
                else:
                    param.grad += (adapted_parameters[name] - param) / self.args.meta_batch_size

    def adapt(self, trn_inputs: torch.Tensor, trn_targets: torch.Tensor, 
              first_order: bool = True) -> dict[str, any]:
        named_params = OrderedDict(self.meta_model.named_parameters())

        task_iter = self.args.task_iter_trn if self.is_meta_training() else self.args.task_iter_eval
        for _ in range(task_iter):
            trn_logits = functional_call(self.base_model, named_params, trn_inputs)
            trn_loss = self.nll(trn_logits, trn_targets)
            trn_grads = torch.autograd.grad(trn_loss, named_params.values(), create_graph=False)
            for (name, param), grad in zip(named_params.items(), trn_grads):
                named_params[name] = param - self.args.task_lr * grad

        return {'named_params': named_params}
