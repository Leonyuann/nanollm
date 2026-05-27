""" Optimizer implementations.
This module contains implementations of optimization algorithms and learning rate schedules commonly used in
training deep learning models. The optimizers include stochastic gradient descent (SGD) and AdamW, while the 
learning rate schedule is a cosine annealing schedule with linear warmup. Additionally, a function for 
gradient clipping is provided to prevent exploding gradients during training.

Classes:
    SGD: A simple implementation of stochastic gradient descent optimizer.
    AdamW: A simple implementation of AdamW optimizer without bias correction.

Functions:
    cosine_learning_lr_schedule: Computes the learning rate at a given iteration using a cosine annealing
        schedule with linear warmup.
    gradient_clipping: Clips the gradients of the given parameters to have a maximum L2 norm.
"""
from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

from config_manager import load_AdamWConfig

adamw_config = load_AdamWConfig()

class SGD(torch.optim.Optimizer):
    """
    A simple implementation of stochastic gradient descent (SGD) optimizer.
    Args:
        params: An iterable of parameters to optimize or dicts defining parameter groups.
        lr: Learning rate (default: 1e-3).      
    Raises:
        ValueError: If the learning rate is negative.
    """
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
            t = state.get("t", 0) # Get iteration number from the state, or 0.
            grad = p.grad.data # Get the gradient of loss with respect to p.
            p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
            state["t"] = t + 1 # Increment iteration number.
        return loss
        
class AdamW(torch.optim.Optimizer):
    """ A simple implementation of AdamW optimizer without bias correction.
    Args:
        params: An iterable of parameters to optimize or dicts defining parameter groups.
        lr: Learning rate (default: 1e-3).
        betas: Coefficients used for computing running averages of gradient and its square (default
            (0.9, 0.95)).
        eps: Term added to the denominator to improve numerical stability (default: 1e-8).
        weight_decay: Weight decay (L2 penalty) (default: 1e-8
    Raises:
        ValueError: If any of the hyperparameters are negative.


    """
    def __init__(
        self,
        params,
        lr: float = adamw_config.lr,
        betas: tuple[float, float] = (adamw_config.beta1, adamw_config.beta2),
        eps : float = adamw_config.eps,
        weight_decay: float = adamw_config.weight_decay
    ):
        if lr < 0 or betas[0] < 0 or betas[1] < 0 or eps < 0 or weight_decay < 0:
            raise ValueError(
                f"Invalid hyperparameter value: lr={lr}, beta1={betas[0]}, "
                f"beta2={betas[1]}, eps={eps}, weight_decay={weight_decay}"
            )
        
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay
        }

        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        # Iterate all group in param_groups    
        for group in self.param_groups:

            # Extract hyper-parameter for current group
            lr = group["lr"]
            beta1 = group["betas"][0]
            beta2 = group["betas"][1]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            # Iterate all parameters in this group
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                # Get t, m, v if exist
                state = self.state[p]
                t = state.get("step",1)
                m = state.get("first_moment",torch.zeros_like(p))
                v = state.get("second_moment",torch.zeros_like(p))

                # Weigth decay
                grad = p.grad.data
                lr_t = lr * math.sqrt(1 - beta2 ** (t)) / (1 - beta1 ** (t))
                p.data = p.data - lr * weight_decay * p.data

                # update fist moment esitmate and second raw moment estimate
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad * grad)
                
                # update state
                state["step"] = t + 1
                state["first_moment"] = m
                state["second_moment"] = v

                # update parameter
                p.data = p.data - lr_t * m / (torch.sqrt(v) + eps)

        return loss


def cosine_learning_lr_schedule(
    step: int,
    max_lr: float,
    min_lr: float,
    t_w: int,
    t_c: int,
) -> float:
    """
    Cosine anealing learning rate schedule with linear warmup.

    Args:
        step: Current iteration number (must be non-negative).
        max_lr: Maximum learning rate (must be positive).
        min_lr: Minimum learning rate (must be non-negative).
        t_w: Number of warmup iterations (must be non-negative).    
        t_c: Number of iterations in one cosine cycle (must be non-negative).
    """
    if step < 0 :
        raise ValueError(f"lr_schedule: Invalid parameter value: step = {step}")
    if max_lr <= 0 :                                    
        raise ValueError(f"lr_schedule: Invalid parameter value: max_lr = {max_lr}")
    if min_lr < 0 :                                                             
        raise ValueError(f"lr_schedule: Invalid parameter value: min_lr = {min_lr}")
    if t_w < 0 :
        raise ValueError(f"lr_schedule: Invalid parameter value: t_w = {t_w}")
    if t_c < 0 :
        raise ValueError(f"lr_schedule: Invalid parameter value: t_c = {t_c}")
    
    if step < t_w :
        lr = max_lr * step / t_w
    elif step <= t_c :
        lr = min_lr + (1/2) * (max_lr - min_lr) * (1 + math.cos(math.pi * (step - t_w) / (t_c - t_w)))
    else :
        lr = min_lr 
    return lr   

def gradient_clipping(parameters: Iterable[torch.nn.Parameter] , max_l2_norm: float) -> None:
    """
    Clips the gradients of the given parameters to have a maximum L2 norm.

    Args:
        parameters: An iterable of parameters whose gradients will be clipped.
        max_l2_norm: The maximum allowed L2 norm of the all-sumed gradients. Must be non-negative.
    Raises:
        ValueError: If `max_l2_norm` is negative.
    """
    assert max_l2_norm >= 0, f"Invalid parameter value: max_l2_norm = {max_l2_norm}"

    eps = 1e-6

    total_l2_norm_squre = 0.0
    for para in parameters :
        if para.grad is None:
            continue

        total_l2_norm_squre += torch.sum(para.grad.pow(2))

    total_l2_norm = torch.sqrt(total_l2_norm_squre)
    if total_l2_norm <= max_l2_norm :
        return
    
    scale = max_l2_norm /(total_l2_norm + eps)
    for para in parameters :
        if para.grad is None :
            continue
        para.grad.mul_(scale)

    return






