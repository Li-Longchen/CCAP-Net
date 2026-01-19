# -*- coding: utf-8 -*-
"""
Conditional Domain Adversarial Module
"""

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Function
from typing import Optional, Tuple, Any


class GradientReverseFunction(Function):
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: float = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        return input * 1.0

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None


class WarmStartGradientReverseLayer(nn.Module):

    def __init__(self, alpha: float = 1.0, lo: float = 0.0, hi: float = 1.,
                 max_iters: int = 1000, auto_step: bool = False):
        super().__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        coeff = float(
            2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo
        )
        if self.auto_step:
            self.step()
        return GradientReverseFunction.apply(input, coeff)

    def step(self):
        self.iter_num += 1


def binary_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    with torch.no_grad():
        batch_size = target.size(0)
        pred = (output >= 0.5).float().t().view(-1)
        correct = pred.eq(target.view(-1)).float().sum()
        correct.mul_(100. / batch_size)
        return correct


class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, 1)
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        x = self.dropout(torch.relu(self.fc1(x)))
        return torch.sigmoid(self.fc2(x))


class DomainAdversarialLoss(nn.Module):

    def __init__(self, domain_discriminator: nn.Module, max_iter: int = 1000):
        super().__init__()
        self.grl = WarmStartGradientReverseLayer(alpha=10.0, lo=0.0, hi=1.0,
                                                 max_iters=max_iter, auto_step=True)
        self.domain_discriminator = domain_discriminator
        self.bce = nn.BCELoss()

    def forward(self, f_s: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
        f = self.grl(torch.cat((f_s, f_t), dim=0))
        d = self.domain_discriminator(f)
        d_s, d_t = d.chunk(2, dim=0)
        lbl_s = torch.ones((f_s.size(0), 1)).to(f_s.device)
        lbl_t = torch.zeros((f_t.size(0), 1)).to(f_t.device)
        return 0.5 * (self.bce(d_s, lbl_s) + self.bce(d_t, lbl_t))


class ConditionalDomainDiscriminator(nn.Module):
    """CDAN"""

    def __init__(self, in_dim: int = 192, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class ConditionalDomainAdversarialLoss(nn.Module):
    """CDAN loss"""

    def __init__(self, domain_discriminator: nn.Module, max_iter: int = 1000):
        super().__init__()
        self.grl = WarmStartGradientReverseLayer(alpha=10.0, lo=0.0, hi=1.0,
                                                 max_iters=max_iter, auto_step=True)
        self.domain_discriminator = domain_discriminator
        self.bce = nn.BCELoss()

    def forward(self, f_s: torch.Tensor, y_s: torch.Tensor,
                f_t: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
        def tensor_aggregate(features, class_probs):
            B, f_dim = features.shape
            _, C = class_probs.shape
            outer = torch.bmm(class_probs.unsqueeze(2), features.unsqueeze(1))
            return outer.view(B, -1)

        agg_s = tensor_aggregate(f_s, y_s)
        agg_t = tensor_aggregate(f_t, y_t)

        combined = self.grl(torch.cat([agg_s, agg_t], dim=0))
        out = self.domain_discriminator(combined)

        B_s, B_t = f_s.size(0), f_t.size(0)
        d_s, d_t = out[:B_s], out[B_s:]
        lbl_s = torch.ones((B_s, 1), device=f_s.device)
        lbl_t = torch.zeros((B_t, 1), device=f_t.device)

        return 0.5 * (self.bce(d_s, lbl_s) + self.bce(d_t, lbl_t))
