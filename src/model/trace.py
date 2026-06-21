import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

from .Base.utils import infoNCE_align_loss, kl_divergence


class GradientReverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, lambda_: float):
        ctx.lambda_ = lambda_
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_ * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self, lambda_: float = 1.0):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReverseFunction.apply(x, self.lambda_)


class ASRRefiner(nn.Module):
    """
    Anti-Spurious Representation Refiner (ASR-Refiner)
    """
    def __init__(self, fea_dim: int = 256, rho: float = 0.95, epsilon: float = 1e-8, max_events: int = 1000):
        super(ASRRefiner, self).__init__()
        self.fea_dim = fea_dim
        self.rho = rho
        self.epsilon = epsilon

        self.register_buffer('mu_ej', torch.zeros(max_events, fea_dim))
        self.register_buffer('sigma2_ej', torch.ones(max_events, fea_dim))
        self.register_buffer('mu_dot_j', torch.zeros(fea_dim))
        self.register_buffer('event_counts', torch.zeros(max_events, dtype=torch.long))

        self.gate_alpha = nn.Parameter(torch.ones(fea_dim))
        self.gate_beta = nn.Parameter(torch.zeros(fea_dim))

    @torch.no_grad()
    def update_statistics(self, H_2d: torch.Tensor, events: torch.Tensor):
        unique_events = torch.unique(events)
        for e in unique_events.tolist():
            mask = (events == e)
            samples = H_2d[mask]
            n_e = samples.size(0)
            if n_e == 0:
                continue
            mu_batch = samples.mean(dim=0)
            var_batch = samples.var(dim=0, unbiased=True) if n_e > 1 else torch.zeros_like(mu_batch)
            if self.event_counts[e] == 0:
                self.mu_ej[e] = mu_batch 
                self.sigma2_ej[e] = var_batch
            else:
                self.mu_ej[e] = self.rho * self.mu_ej[e] + (1 - self.rho) * mu_batch
                self.sigma2_ej[e] = self.rho * self.sigma2_ej[e] + (1 - self.rho) * var_batch
            self.event_counts[e] += n_e

        mu_dot_batch = H_2d.mean(dim=0)
        if self.event_counts.sum() == 0:
            self.mu_dot_j = mu_dot_batch
        else:
            self.mu_dot_j = self.rho * self.mu_dot_j + (1 - self.rho) * mu_dot_batch

    def compute_sensitivity_scores(self, H_2d: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
        S = torch.zeros_like(H_2d)
        for i in range(H_2d.size(0)):
            e = int(events[i].item())
            if self.event_counts[e] > 0:
                mu_ej = self.mu_ej[e].detach()
                sig2_ej = self.sigma2_ej[e].detach()
                mu_dot = self.mu_dot_j.detach()
                numerator = (mu_ej - mu_dot) ** 2
                denominator = (H_2d[i] - mu_ej) ** 2 + sig2_ej + self.epsilon
                S[i] = numerator / denominator
        return S

    def forward(self, H: torch.Tensor, events: torch.Tensor):
        original_shape = H.shape
        is_seq = (H.dim() == 3)
        H_2d = H.mean(dim=1) if is_seq else H

        self.update_statistics(H_2d, events)
        S = self.compute_sensitivity_scores(H_2d, events)
        S_norm = F.normalize(S, p=2, dim=1)

        gate_input = self.gate_alpha * S_norm + self.gate_beta
        m_2d = torch.clamp(gate_input, 0.0, 1.0)

        if is_seq:
            m = m_2d.unsqueeze(1).expand(-1, original_shape[1], -1)
        else:
            m = m_2d

        Z = H * (1 - m)
        return Z, S, m


class EventConfounderHead(nn.Module):
    def __init__(self, input_dim: int, num_events: int, lambda_grl: float = 1.0, hidden_dim: int = 256, dropout: float = 0.5):
        super(EventConfounderHead, self).__init__()
        self.grl = GradientReversal(lambda_=lambda_grl)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_events)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.grl(features)
        logits = self.classifier(x)
        return logits


def compute_iepa_loss(logits: torch.Tensor, labels: torch.Tensor, events: torch.Tensor, alpha: float = 1.0, num_classes: Optional[int] = None) -> torch.Tensor:
    """
    Compute the KL alignment loss of Intra-Event Prediction Aligner (IEPA)
    """
    if num_classes is None:
        num_classes = logits.size(-1)
    probs = F.softmax(logits, dim=-1)
    unique_events = torch.unique(events)
    total_n = float(events.numel())
    device = logits.device
    iepa_total = torch.tensor(0.0, device=device)

    for e in unique_events.tolist():
        mask = (events == e)
        n_e = int(mask.sum().item())
        if n_e == 0:
            continue
        labels_e = labels[mask]
        counts_e = torch.bincount(labels_e, minlength=num_classes).to(logits.dtype)
        p_hat_e = (counts_e + alpha) / (n_e + alpha * num_classes)
        bar_p_e = probs[mask].mean(dim=0)

        p_logits = torch.log(bar_p_e + 1e-8).unsqueeze(0)
        q_logits = torch.log(p_hat_e + 1e-8).unsqueeze(0)
        kl_e = kl_divergence(p_logits, q_logits)
        w_e = n_e / total_n
        iepa_total = iepa_total + (w_e * kl_e)

    return iepa_total


def compute_asr_align_loss(unrefined_list: List[torch.Tensor], refined_list: List[torch.Tensor], weight: float = 1.0) -> torch.Tensor:

    assert len(unrefined_list) == len(refined_list), "assert len(unrefined_list) == len(refined_list)"
    loss = torch.tensor(0.0, device=unrefined_list[0].device)
    for u, r in zip(unrefined_list, refined_list):
        if u.dim() == 3:
            u = u.mean(dim=1)
        if r.dim() == 3:
            r = r.mean(dim=1)
        loss = loss + infoNCE_align_loss(r, u)
    return loss * weight




