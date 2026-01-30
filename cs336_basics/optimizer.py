import torch
from torch import Tensor
from jaxtyping import Float, Integer
from collections.abc import Callable, Iterable
from typing import Optional
import math


def cross_entropy(o: Float[Tensor, "... sequence_length vocab_size"], targets: Integer[Tensor, "... sequence_length"]):
    """
    Returns output of dimension ... (single batch_num number)
    """
    max_elems = torch.max(o, dim=-1, keepdim=True).values
    o = o - max_elems
    logsumexpd = o.exp().sum(dim=-1).log()
    selected_logits = o.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    neg_log_probs = -selected_logits + logsumexpd
    return neg_log_probs.mean()


def learning_rate_schedule(t: int, alpha_max: float, alpha_min: float, Tw: int, Tc: int):
    return ((t < Tw) * t/Tw * alpha_max +
            (Tw <= t <= Tc) * (alpha_min + (alpha_max - alpha_min)*0.5*(1 + math.cos((t - Tw)/(Tc - Tw)*math.pi)))+
            (Tc < t) * alpha_min)


def gradient_clipping(params: Iterable[torch.nn.Parameter], max_norm: float, eps=1e-06):
    grad_norm = math.sqrt(sum((p.grad.data**2).sum() for p in params if p.grad is not None))
    if grad_norm >= max_norm:
        for p in params:
            if p.grad is not None:
                p.grad.data.mul_(max_norm/(grad_norm + eps))


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.
        return loss


class AdamW(torch.optim.Optimizer):
    """
    AdamW optimizer: implemented according to Loshchilov and Hutter 2019.
    For each parameter keeps the first and second moment, and applies weight decay independent of the gradient update.
    """

    def __init__(self, params, lr, weight_decay, betas, eps):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if betas[0] < 0 or betas[1] < 0:
            raise ValueError("Beta values (parameters b1, b2) must be positive.")
        if eps < 0 or eps >= 1:
            raise ValueError("Epsilon offset out of allowed range [0, 1).")
        if weight_decay < 0:
            raise ValueError("Weight decay parameter (lam) must be positive.")

        defaults = {'b1': betas[0], 'b2': betas[1],
                    'lr': lr,
                    'eps': eps, 'lam': weight_decay}

        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable]=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            b1 = group['b1']
            b2 = group['b2']
            eps = group['eps']
            lam = group['lam']
            alpha = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                grad = p.grad.data
                m = state.get("m", torch.zeros_like(grad))
                v = state.get("v", torch.zeros_like(grad))
                t = state.get("t", 1)

                m = b1 * m + (1 - b1) * grad
                v = b2 * v + (1 - b2) * (grad**2)
                alpha_t = alpha * math.sqrt(1 - b2**t)/(1 - b1**t)

                p.data -= alpha_t * m / (torch.sqrt(v) + eps)
                p.data -= alpha * lam * p.data

                state["m"] = m
                state["v"] = v
                state["t"] = t + 1

        return loss
