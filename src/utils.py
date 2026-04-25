import torch


class Checkpointer:
    def __init__(self, save_fn: callable, alg_name: str) -> None:
        self.save_fn = save_fn
        self.alg_name = alg_name
        self.counter = 0
        self.best_acc = 0

    def update(self, acc: float) -> None:
        self.counter += 1
        self.save_fn(self.alg_name + f'_{self.counter:02d}.ct')

        if acc > self.best_acc:
            self.best_acc = acc
            self.save_fn(self.alg_name + '_final.ct')


# Modified from https://docs.backpack.pt/en/master/use_cases/example_cg_newton.html
@torch.no_grad()
def cg(A, b, x0=None, maxiter=None, tol=1e-5, atol=1e-8):
        r"""Solve :math:`Ax = b` for :math:`x` using conjugate gradient.

        The interface is similar to CG provided by :code:`scipy.sparse.linalg.cg`.

        The main iteration loop follows the pseudo code from Wikipedia:
            https://en.wikipedia.org/w/index.php?title=Conjugate_gradient_method&oldid=855450922

        Parameters
        ----------
        A : function
            Function implementing matrix-vector multiplication by `A`.
        b : torch.Tensor
            Right-hand side of the linear system.
        x0 : torch.Tensor
            Initialization estimate.
        maxiter: int
            Maximum number of iterations.
        tol: float
            Relative tolerance to accept convergence. Stop if
            :math:`|| A x - b || / || b || <` `tol`.
        atol: float
            Absolute tolerance to accept convergence. Stop if
            :math:`|| A x - b || <` `atol`

        Returns
        -------
        x (torch.Tensor): Approximate solution
            :math:`x` of the linear system
        """
        maxiter = b.numel() if maxiter is None else min(maxiter, b.numel())
        x = torch.zeros_like(b, requires_grad=False) if x0 is None else x0.detach()

        # initialize parameters
        r = b - A(x)
        p = r.clone()
        rs_old = (r**2).sum()

        # iterate
        iterations = 0
        while True:
            Ap = A(p)
            alpha = rs_old / (p @ Ap)

            x += alpha * p
            r -= alpha * Ap
            rs_new = (r**2).sum()
            iterations += 1

            # if iterations > maxiter
            if iterations >= maxiter:
                return x

            p *= rs_new / rs_old
            p += r
            rs_old = rs_new


def R_op(ys, xs, vs):
    if isinstance(ys, tuple):
        ws = [torch.zeros_like(y, requires_grad=True) for y in ys]
    else:
        ws = torch.zeros_like(ys, requires_grad=True)

    gs = torch.autograd.grad(ys,
                             xs,
                             grad_outputs=ws,
                             create_graph=True,
                             retain_graph=True,
                             allow_unused=True)

    re = torch.autograd.grad(gs,
                             ws,
                             grad_outputs=vs,
                             create_graph=False,
                             retain_graph=True,
                             allow_unused=True)

    return re


@torch.no_grad()
def L_op(ys, xs, ws, create_graph=False, retain_graph=True, vec=True):
    vJ = torch.autograd.grad(ys,
                             xs,
                             grad_outputs=ws,
                             create_graph=create_graph,
                             retain_graph=retain_graph,
                             allow_unused=True)

    if vec:
        vJ = torch.cat([j.detach().reshape(-1) for j in vJ])

    return vJ
