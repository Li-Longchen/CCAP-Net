# -*- coding: utf-8 -*-
"""
CSCFA Module: Cross-Source Contrastive Feature Alignment

This module implements the CSCFA part described in the paper:
1) Multi-source setting: each non-target subject is treated as an individual source domain S_j.
2) "First align each source domain with the target domain, then align all source domains together":
   - Source-Target alignment (per source domain): MMD + contrastive (Eq. (4) + Eq. (6))
   - Source-Source alignment (across different source domains): contrastive (Eq. (7))
3) Overall CSCFA loss (Eq. (8)): L = L_mmd + α * L_st + β * L_ss
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MMDLoss(nn.Module):
    """Maximum Mean Discrepancy (linear kernel), as Eq. (4) in the paper."""

    def __init__(self, kernel_type: str = 'linear'):
        super().__init__()
        self.kernel_type = kernel_type

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Paper uses linear kernel and focuses on mean difference.
        source_norm = F.normalize(source, p=2, dim=1)
        target_norm = F.normalize(target, p=2, dim=1)
        source_mean = torch.mean(source_norm, dim=0)
        target_mean = torch.mean(target_norm, dim=0)
        return torch.sum((source_mean - target_mean) ** 2)


class ContrastiveLoss(nn.Module):
    """Contrastive losses used in CSCFA (Eq. (6) & Eq. (7))."""

    def __init__(self, temperature: float = 0.5, similarity_threshold: float = 0.7):
        super().__init__()
        self.temperature = temperature
        self.similarity_threshold = similarity_threshold

    @staticmethod
    def cosine_similarity_matrix(f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        f1_norm = F.normalize(f1, p=2, dim=1)
        f2_norm = F.normalize(f2, p=2, dim=1)
        return torch.mm(f1_norm, f2_norm.t())

    def source_target_contrastive_loss(self,
                                       source_features: torch.Tensor,
                                       target_features: torch.Tensor) -> torch.Tensor:
        """
        Source-Target contrastive loss (Eq. (6)).
        Target is unlabeled -> use similarity threshold to define positives.
        """
        if source_features.numel() == 0 or target_features.numel() == 0:
            return torch.tensor(0.0, device=source_features.device)

        sim_matrix = self.cosine_similarity_matrix(source_features, target_features)

        # Fixed threshold from paper; add a safe fallback to avoid empty positives.
        threshold = self.similarity_threshold

        positive_mask = (sim_matrix > threshold).float()

        # Ensure each source sample has at least one positive to avoid NaN.
        max_sim_indices = torch.argmax(sim_matrix, dim=1)
        positive_mask[torch.arange(source_features.size(0), device=source_features.device), max_sim_indices] = 1.0

        sim_scaled = sim_matrix / self.temperature
        sim_max = sim_scaled.max(dim=1, keepdim=True)[0]
        exp_sim = torch.exp(sim_scaled - sim_max)

        positive_sim = (exp_sim * positive_mask).sum(dim=1)
        all_sim = exp_sim.sum(dim=1)

        loss = -torch.log(positive_sim / (all_sim + 1e-8) + 1e-8)
        return torch.clamp(loss, max=10.0).mean()

    def source_source_contrastive_loss(self,
                                       source_features: torch.Tensor,
                                       source_labels: torch.Tensor,
                                       source_domains: torch.Tensor) -> torch.Tensor:
        """
        Cross-source (source-source) contrastive loss (Eq. (7)).
        Only considers pairs from *different* source domains (domains !=).
        Positives: same label + different domain
        Negatives: different label + different domain
        """
        B = source_features.size(0)
        if B < 2:
            return torch.tensor(0.0, device=source_features.device)

        # Normalize features for cosine similarity
        f = F.normalize(source_features, p=2, dim=1)
        sim = torch.mm(f, f.t())  # (B, B)

        sim_scaled = sim / self.temperature
        sim_max = sim_scaled.max(dim=1, keepdim=True)[0]
        exp_sim = torch.exp(sim_scaled - sim_max)

        # Masks
        device = source_features.device
        domains = source_domains.view(-1)
        labels = source_labels.view(-1)

        domain_diff = domains.unsqueeze(0) != domains.unsqueeze(1)
        not_self = ~torch.eye(B, device=device, dtype=torch.bool)
        valid_pair = domain_diff & not_self

        if valid_pair.sum() == 0:
            return torch.tensor(0.0, device=device)

        label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
        pos_mask = valid_pair & label_eq
        denom_mask = valid_pair

        numerator = (exp_sim * pos_mask.float()).sum(dim=1)
        denominator = (exp_sim * denom_mask.float()).sum(dim=1)

        valid_anchor = numerator > 0
        if valid_anchor.sum() == 0:
            return torch.tensor(0.0, device=device)

        loss = -torch.log(numerator[valid_anchor] / (denominator[valid_anchor] + 1e-8) + 1e-8)
        return torch.clamp(loss, max=10.0).mean()


class CSCFAModule(nn.Module):
    """
    CSCFA module wrapper.

    Inputs:
        source_features: (B_s, D)
        target_features: (B_t, D)
        source_labels  : (B_s,) label indices (int64)
        source_domains : (B_s,) source domain (subject) indices (int64)

    Returns:
        total_loss, (mmd_loss, st_contrast_loss, ss_contrast_loss)
    """

    def __init__(self,
                 feature_dim: int = 64,
                 temperature: float = 0.5,
                 similarity_threshold: float = 0.7):
        super().__init__()
        self.mmd_loss = MMDLoss(kernel_type='linear')
        self.contrastive = ContrastiveLoss(temperature=temperature,
                                           similarity_threshold=similarity_threshold)

    def forward(self,
                source_features: torch.Tensor,
                target_features: torch.Tensor,
                source_labels: torch.Tensor,
                source_domains: torch.Tensor,
                alpha: float = 1.0,
                beta: float = 1.0) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Compute CSCFA loss (Eq. (8)):
            L = L_mmd + α * L_st + β * L_ss

        - L_mmd and L_st are computed by aligning each source domain with the target domain individually.
        - L_ss is computed across different source domains.
        """
        device = source_features.device
        total_mmd = torch.tensor(0.0, device=device)
        total_st_contrast = torch.tensor(0.0, device=device)

        # Per-source-domain source-target alignment
        unique_domains = torch.unique(source_domains)
        n_domains = 0
        for d in unique_domains:
            idx = source_domains == d
            f_sd = source_features[idx]
            if f_sd.size(0) == 0:
                continue

            n_domains += 1

            # MMD between this source domain and target
            total_mmd = total_mmd + self.mmd_loss(f_sd, target_features)

            # Contrastive between this source domain and target
            total_st_contrast = total_st_contrast + self.contrastive.source_target_contrastive_loss(f_sd, target_features)


        if n_domains > 0:
            total_mmd = total_mmd / float(n_domains)
            total_st_contrast = total_st_contrast / float(n_domains)

        # Cross-source alignment among all source domains
        ss_contrast = self.contrastive.source_source_contrastive_loss(source_features, source_labels, source_domains)

        total = total_mmd + alpha * total_st_contrast + beta * ss_contrast
        return total, (total_mmd, total_st_contrast, ss_contrast)
