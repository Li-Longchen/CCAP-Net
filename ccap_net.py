# -*- coding: utf-8 -*-
"""
CCAP-Net Model


Main modules used by the training script:
- Feature Extractor: 310 -> 128 -> 64 (dropout=0.1)
- PIL (Prototype Interactive Learning):
    * prototypes computed from labeled source features (Eq. 13)
    * bilinear transform with trainable matrix U (Eq. 14)
    * softmax to get prototype-interactive features (Eq. 15)
    * pairwise learning losses are computed in train.py
- The CSCFA/CCSA losses are implemented in cscfa.py / cdan.py and called in train.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import normalized_mutual_info_score


class FeatureExtractor(nn.Module):
    """Feature extractor: 310 -> 128 -> 64 (dropout=0.1)"""

    def __init__(self, input_dim: int = 310, hidden_dim: int = 128, output_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        return x

    def forward_with_variance(self, x: torch.Tensor):
        """Forward + feature variance (optional stability monitor)."""
        x = self.forward(x)
        feature_var = torch.var(x, dim=0).mean()
        return x, feature_var

    def get_parameters(self):
        return [{'params': self.parameters()}]


class CCAPNet(nn.Module):
    """CCAP-Net"""

    def __init__(self,
                 input_dim: int = 310,
                 hidden_dim: int = 128,
                 feature_dim: int = 64,
                 num_classes: int = 3,
                 max_iter: int = 1200,
                 upper_threshold: float = 0.9,
                 lower_threshold: float = 0.5):
        super().__init__()

        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.max_iter = max_iter

        # Thresholds used for target-pair selection in PIL
        self.upper_threshold = float(upper_threshold)
        self.lower_threshold = float(lower_threshold)
        self.threshold = float(upper_threshold)  # dynamic threshold used for pseudo pair labels

        self.feature_extractor = FeatureExtractor(input_dim, hidden_dim, feature_dim)

        # PIL parameters
        # U: bilinear transform matrix (Eq. 14). Initialized as identity for stable start.
        self.U = nn.Parameter(torch.eye(feature_dim))
        # stored_mat: prototype matrix (feature_dim x num_classes), updated by cluster_label_update()
        # In the paper (Eq. 13), prototypes are computed from labeled source features (not learned by gradient).
        self.register_buffer('stored_mat', torch.randn(feature_dim, num_classes))

        # Will be assigned externally (train.py)
        self.cdan_loss_fn = None

        # Optional: track feature variance
        self.feature_variance = torch.tensor(0.0)

    def forward(self,
                source: torch.Tensor,
                target: torch.Tensor) -> tuple:
        """
        Returns:
            logits_s: (B_s, C)
            prob_s  : (B_s, C)  (prototype-interactive features, Eq. 15)
            feat_s  : (B_s, D)  (domain-invariant features)
            feat_t  : (B_t, D)
            sim_s   : (B_s, B_s) predicted pair-sim (continuous in [0,1])
            sim_t   : (B_t, B_t) predicted pair-sim (continuous in [0,1])
            logits_t: (B_t, C)
            prob_t  : (B_t, C)
        """
        feat_s, var_s = self.feature_extractor.forward_with_variance(source)
        feat_t, var_t = self.feature_extractor.forward_with_variance(target)
        self.feature_variance = (var_s + var_t) / 2.0

        # Normalize features for prototype interaction
        feat_s_n = F.normalize(feat_s, p=2, dim=1)
        feat_t_n = F.normalize(feat_t, p=2, dim=1)

        # Normalize prototypes to avoid magnitude bias
        proto = F.normalize(self.stored_mat, p=2, dim=0)

        logits_s = torch.matmul(torch.matmul(self.U, feat_s_n.T).T, proto)
        logits_t = torch.matmul(torch.matmul(self.U, feat_t_n.T).T, proto)

        prob_s = F.softmax(logits_s, dim=1)
        prob_t = F.softmax(logits_t, dim=1)

        # PIL pairwise predicted similarity in [0,1]:
        #   p_ij = sum_c p_ic * p_jc
        sim_s = torch.mm(prob_s, prob_s.t()).clamp(0.0, 1.0)
        sim_t = torch.mm(prob_t, prob_t.t()).clamp(0.0, 1.0)

        return logits_s, prob_s, feat_s_n, feat_t_n, sim_s, sim_t, logits_t, prob_t

    @staticmethod
    def get_pairwise_same_label(labels_onehot: torch.Tensor) -> torch.Tensor:
        """y_ij = 1 if same class else 0, from one-hot labels."""
        label_indices = torch.argmax(labels_onehot, dim=1)
        return (label_indices.unsqueeze(0) == label_indices.unsqueeze(1)).float()

    def get_pairwise_by_threshold(self, sim_matrix: torch.Tensor) -> torch.Tensor:
        """Pseudo pair labels for target: 1 if sim > threshold else 0."""
        return (sim_matrix > self.threshold).float()

    def compute_indicator(self, sim_matrix: torch.Tensor):
        """
        Indicator mask for target pairs:
          - 0 for uncertain pairs in (lower, upper)
          - 1 for confident pairs <=lower or >=upper
        """
        indicator = torch.ones_like(sim_matrix)
        mask = (sim_matrix > self.lower_threshold) & (sim_matrix < self.upper_threshold)
        indicator[mask] = 0.0
        return indicator, indicator.sum()

    def update_threshold(self, epoch: int):
        """Linearly decay threshold from upper -> lower over training."""
        progress = float(epoch) / float(self.max_iter)
        self.threshold = self.upper_threshold - progress * (self.upper_threshold - self.lower_threshold)

    def cluster_label_update(self, source_features: torch.Tensor, source_labels: torch.Tensor):
        """
        Update class prototypes using labeled source data (Eq. 13).
        stored_mat[:, c] = mean_{i in class c}(feat_i)
        """
        self.eval()
        with torch.no_grad():
            feat = self.feature_extractor(source_features)
            feat = F.normalize(feat, p=2, dim=1)

            labels_idx = torch.argmax(source_labels, dim=1).cpu().numpy()

            for c in range(self.num_classes):
                mask = (labels_idx == c)
                if mask.sum() > 0:
                    centroid = feat[mask].mean(dim=0)
                    centroid = F.normalize(centroid, p=2, dim=0)
                    self.stored_mat.data[:, c] = centroid

            # Evaluate on source (for logging)
            proto = F.normalize(self.stored_mat, p=2, dim=0)
            logits = torch.matmul(torch.matmul(self.U, feat.T).T, proto)
            pred_labels = torch.argmax(logits, dim=1).cpu().numpy()

            accuracy = (pred_labels == labels_idx).mean()
            nmi = normalized_mutual_info_score(labels_idx, pred_labels)

        return accuracy, nmi

    def evaluate(self, target_features: torch.Tensor, target_labels: torch.Tensor):
        """Evaluate on target domain."""
        self.eval()
        with torch.no_grad():
            feat = self.feature_extractor(target_features)
            feat = F.normalize(feat, p=2, dim=1)
            proto = F.normalize(self.stored_mat, p=2, dim=0)
            logits = torch.matmul(torch.matmul(self.U, feat.T).T, proto)
            pred_labels = torch.argmax(logits, dim=1).cpu().numpy()

            labels_idx = torch.argmax(target_labels, dim=1).cpu().numpy()
            accuracy = (pred_labels == labels_idx).mean()
            nmi = normalized_mutual_info_score(labels_idx, pred_labels)

        return accuracy, nmi

    def get_parameters(self):
        return (self.feature_extractor.get_parameters() + [{'params': [self.U]}])


def weight_init(m):
    """Weight initialization (linear layers only)."""
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.03)
        if m.bias is not None:
            m.bias.data.zero_()
