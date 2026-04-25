from collections import OrderedDict
import torch
import torch.nn as nn
from torch.func import functional_call
from src.meta_alg_base import MetaLearningAlgBase
from src.utils import cg, L_op


class iMAML(MetaLearningAlgBase):
    def __init__(self, args) -> None:
        super().__init__(args)
        assert 0 <= self.args.trunc <= self.args.task_iter_trn

    def _get_meta_model(self, **kwargs) -> nn.Module:
        return self._get_base_model(**kwargs)

    def adapt(self, trn_inputs: torch.Tensor, trn_targets: torch.Tensor, 
              first_order: bool = False) -> dict[str, any]:
        named_params = OrderedDict(self.meta_model.named_parameters())

        task_iter = self.args.task_iter_trn if self.is_meta_training() else self.args.task_iter_eval
        for _ in range(task_iter):
            trn_logits = functional_call(self.base_model, named_params, trn_inputs)
            trn_loss = self.nll(trn_logits, trn_targets)
            trn_grads = torch.autograd.grad(trn_loss, 
                                            named_params.values(), 
                                            create_graph=False)
            for (name, param), grad in zip(named_params.items(), trn_grads):
                named_params[name] = param - self.args.task_lr * grad
        
        if not first_order:
            trn_logits = functional_call(self.base_model, named_params, trn_inputs)
            trn_loss = self.nll(trn_logits, trn_targets)
            trn_grads = torch.autograd.grad(trn_loss, 
                                            named_params.values(), 
                                            create_graph=True)

        return {'named_params': named_params, 'trn_grads': trn_grads}
    
    def meta_backward(self, meta_loss: torch.Tensor, adapted_kwargs: dict[str, any]) -> None:
        trn_grad_vec = nn.utils.parameters_to_vector(adapted_kwargs['trn_grads'])
        
        with torch.no_grad():
            val_grad_vec = torch.autograd.grad(meta_loss, 
                                            adapted_kwargs['named_params'].values(), 
                                            create_graph=False)
            val_grads_vec = nn.utils.parameters_to_vector(val_grad_vec)
            
            def Avec_fn(vec):
                Avec = L_op(trn_grad_vec, 
                            adapted_kwargs['named_params'].values(),
                            vec)
                Avec *= 1 / self.args.lambd
                Avec += vec

                return Avec
            
            meta_grad_vec = cg(A=Avec_fn, 
                            b=val_grads_vec, 
                            x0=val_grads_vec, 
                            maxiter=self.args.trunc)

            offset = 0
            for param in self.meta_model.parameters():
                offset_new = offset + param.numel()
                if param.grad is None:
                    param.grad = meta_grad_vec[offset:offset_new].view_as(param)
                else:
                    param.grad += meta_grad_vec[offset:offset_new].view_as(param)
                offset = offset_new
