from collections import OrderedDict
import torch
import torch.nn as nn
import torch.func as func
from src.meta_alg_base import MetaLearningAlgBase


class BinomMAML(MetaLearningAlgBase):
    def __init__(self, args) -> None:
        super().__init__(args)
        assert 0 <= self.args.trunc <= self.args.task_iter_trn

    def _get_meta_model(self, **kwargs) -> nn.Module:
        return self._get_base_model(**kwargs)
    
    def _get_parallel_hvp_func(self, trn_inputs: torch.Tensor, trn_targets: torch.Tensor) -> callable:
        # In the current version of PyTorch, `autograd.grad` does not support `vmap`
        # `func.grad` needs to be called to compute the gradients again (which have been already computed by torch.autograd.grad in `adapt` method)
        # As a result, backward requires two gradient computations (func.grad + func.vjp) instead of one
        # This doubles the backward computation time
        grad_func = func.grad(
            lambda named_params: self.nll(func.functional_call(self.base_model, 
                                                               named_params, 
                                                               trn_inputs), trn_targets)
            )
        def hvp_func(named_params, named_vecs):
            _, vjp_fn = func.vjp(grad_func, named_params)
            return vjp_fn(named_vecs)[0]
    
        return torch.vmap(hvp_func, in_dims=0)    # better to use pmap, which is not yet available in pytorch; see https://github.com/pytorch/pytorch/issues/129459

    def adapt(self, trn_inputs: torch.Tensor, trn_targets: torch.Tensor, 
              first_order: bool = False) -> dict[str, any]:
        named_params = OrderedDict(self.meta_model.named_parameters())
        if not first_order:
            named_params_trace = OrderedDict({name: [] for name in named_params})
        meta_training = self.is_meta_training()

        task_iter = self.args.task_iter_trn if meta_training else self.args.task_iter_eval
        for _ in range(task_iter):
            if not first_order and meta_training:
                for name, param in named_params.items():
                    named_params_trace[name].append(param)
            trn_logits = func.functional_call(self.base_model, 
                                              named_params, 
                                              trn_inputs)
            trn_loss = self.nll(trn_logits, trn_targets)
            with torch.no_grad():
                trn_grads = torch.autograd.grad(trn_loss, 
                                                named_params.values(), 
                                                create_graph=False)
                for (name, param), grad in zip(named_params.items(), trn_grads):
                    named_params[name] = (param - self.args.task_lr * grad).requires_grad_()

        adapted_kwargs = {'named_params': named_params}
        if not first_order and meta_training:
            with torch.no_grad():
                for name, param_trace in named_params_trace.items():
                    named_params_trace[name] = torch.stack(param_trace)
            adapted_kwargs['named_params_trace'] = named_params_trace
            adapted_kwargs['hvp_func'] = self._get_parallel_hvp_func(trn_inputs, trn_targets)

        return adapted_kwargs

    @torch.no_grad()
    def meta_backward(self, meta_loss: torch.Tensor, adapted_kwargs: dict[str, any]) -> None:
        task_iter, trunc = self.args.task_iter_trn, self.args.trunc
        named_batched_grads = torch.autograd.grad(meta_loss, 
                                                  adapted_kwargs['named_params'].values(), 
                                                  create_graph=False)
        named_batched_grads = {name: grad.expand(task_iter-trunc+1, *grad.size()) 
                               for name, grad in zip(adapted_kwargs['named_params'].keys(), named_batched_grads)}

        for trunc_idx in range(self.args.trunc):
            named_batched_params = {name: params[trunc-1-trunc_idx:task_iter-trunc_idx] 
                                    for name, params in adapted_kwargs['named_params_trace'].items()}
            named_batched_hvps = adapted_kwargs['hvp_func'](named_batched_params, named_batched_grads)

            for batched_hvp in named_batched_hvps.values():
                for hvp_idx in range(task_iter-trunc, 0, -1):
                    batched_hvp[hvp_idx-1] += batched_hvp[hvp_idx] 

            for name, batched_grad in named_batched_grads.items():
                named_batched_grads[name] = batched_grad[-1:] - (self.args.task_lr * trunc / task_iter) * named_batched_hvps[name]

        for name, param in self.meta_model.named_parameters():
            if param.grad is None:
                param.grad = named_batched_grads[name][0]
            else:
                param.grad += named_batched_grads[name][0]
