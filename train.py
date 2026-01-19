# -*- coding: utf-8 -*-
"""
CCAP-Net Training Script

This script supports SEED, SEED-IV, ENTER and REFED.

It also supports two experimental protocols:
  - cross-subject: train/test split within a single session (LOSO)
  - cross-session: concatenate multiple sessions per subject, then LOSO
"""

from __future__ import annotations

import argparse
import datetime
import os
import random
import sys
from typing import Dict, Tuple, Optional

import numpy as np
import scipy.io as scio
import torch
import torch.nn.functional as F
import torch.utils.data as Data
from torch.optim import RMSprop

# Ensure local imports work when running as a script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ccap_net import CCAPNet, weight_init
from cdan import (ConditionalDomainAdversarialLoss, ConditionalDomainDiscriminator,
                  DomainAdversarialLoss, DomainDiscriminator)
from cscfa import CSCFAModule
from utils import (
    get_dataset,
    get_dataset_cross_session,
    get_dataset_info,
    list_common_subject_numbers,
    list_subject_numbers,
)


class ExperimentLogger:
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.log_file = open(log_path, 'a', encoding='utf-8')

    def log(self, message: str, print_console: bool = True):
        if print_console:
            print(message)
        self.log_file.write(message + '\n')
        self.log_file.flush()

    def close(self):
        self.log_file.close()


def save_hyperparameter_config(config: Dict, save_path: str):
    """Generate a hyperparameter config file (for reproducibility)."""
    content = f"""======================================================================
CCAP-Net Hyperparameter Configuration
======================================================================

【Network】
----------------------------------------------------------------------
Name                     | Value         | Notes
----------------------------------------------------------------------
input_dim                 | 310           | DE features
hidden_1                  | {config['hidden_1']:<13} | hidden dim
hidden_2 (feature_dim)    | {config['hidden_2']:<13} | feature dim
num_classes               | {config['num_classes']:<13} | #classes
dropout_rate              | 0.1           | dropout

【Training】
----------------------------------------------------------------------
batch_size                | {config['batch_size']:<13}
max_iter (epochs)         | {config['max_iter']:<13}
warm_up_epochs (CDAN)     | {config['warm_up_epochs']:<13}
warm_up_transfer_epochs   | {config.get('warm_up_transfer_epochs', 0):<13} | warm-up before DANN/CSCFA
learning_rate             | {config['lr']:<13}
weight_decay              | {config['weight_decay']:<13}
optimizer                 | RMSprop

【Loss Weights / Params】
----------------------------------------------------------------------
transfer_weight (DANN)    | {config.get('transfer_weight', 0.5):<13}
cscfa_weight              | {config['cscfa_weight']:<13}
cdan_weight               | {config['cdan_weight']:<13}
ce_weight                 | {config.get('ce_weight', 1.0):<13}
pseudo_label_threshold    | {config.get('pseudo_label_threshold', 0.95):<13}

【Thresholds】
----------------------------------------------------------------------
upper_threshold           | {config['upper_threshold']:<13}
lower_threshold           | {config['lower_threshold']:<13}
CSCFA temperature         | 0.5
CSCFA sim_threshold       | 0.7

【Seed】
----------------------------------------------------------------------
random_seed               | {config['seed']:<13}
"""
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(content)


def setup_seed(seed: int = 20):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def lambda_schedule(progress: float) -> float:
    """Eq. (12): λ(p) = 2/(1+exp(-10p)) - 1"""
    return float(2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0)


def train_epoch(model: CCAPNet,
                loader_source: Data.DataLoader,
                loader_target: Data.DataLoader,
                dann_loss: DomainAdversarialLoss,
                cscfa_module: CSCFAModule,
                optimizer: torch.optim.Optimizer,
                epoch: int,
                config: Dict,
                device: torch.device,
                cdan_enabled: bool = False) -> Tuple[float, float, float]:
    """Train for one epoch."""
    model.train()

    num_batches = min(len(loader_source), len(loader_target))
    src_iter = iter(loader_source)
    tgt_iter = iter(loader_target)

    total_loss_sum = 0.0
    pil_loss_sum = 0.0
    trans_loss_sum = 0.0

    # Stats for CDAN pseudo-label coverage (diagnostic)
    cdan_total = 0
    cdan_selected = 0

    warm_transfer = int(config.get('warm_up_transfer_epochs', 0))

    for _ in range(num_batches):
        try:
            x_s, y_s, d_s = next(src_iter)
        except StopIteration:
            src_iter = iter(loader_source)
            x_s, y_s, d_s = next(src_iter)

        try:
            x_t, y_t = next(tgt_iter)
        except StopIteration:
            tgt_iter = iter(loader_target)
            x_t, y_t = next(tgt_iter)

        x_s = x_s.to(device)
        y_s = y_s.to(device)
        d_s = d_s.to(device)
        x_t = x_t.to(device)

        # Forward
        logits_s, prob_s, f_s, f_t, sim_s, sim_t, logits_t, prob_t = model(x_s, x_t)

        # -------------------------
        # PIL losses (Eq. 16-18)
        # -------------------------
        eta = 1e-5

        with torch.no_grad():
            gt_s = model.get_pairwise_same_label(y_s)  # source ground-truth pair labels
            gt_t = model.get_pairwise_by_threshold(sim_t)  # target pseudo pair labels

        # BCE on pairwise similarity
        bce_s = -(torch.log(sim_s + eta) * gt_s) - (1 - gt_s) * torch.log(1 - sim_s + eta)
        bce_t = -(torch.log(sim_t + eta) * gt_t) - (1 - gt_t) * torch.log(1 - sim_t + eta)

        Ls = bce_s.mean()

        indicator, nb_selected = model.compute_indicator(sim_t)
        Lt = torch.sum(indicator * bce_t) / (nb_selected + 1e-8)

        pil_loss = Ls + Lt

        # Standard CE (optional, but keep for stability; uses logits!)
        labels_idx = torch.argmax(y_s, dim=1)
        ce_loss = F.cross_entropy(logits_s, labels_idx)
        pil_loss = pil_loss + config.get('ce_weight', 1.0) * ce_loss        # -------------------------
        # DANN (Eq. 9)
        # -------------------------
        tw = float(config.get('transfer_weight', 0.5))
        min_b = min(f_s.size(0), f_t.size(0))
        perm_s = torch.randperm(f_s.size(0), device=device)
        perm_t = torch.randperm(f_t.size(0), device=device)

        f_s_b = f_s[perm_s][:min_b]
        f_t_b = f_t[perm_t][:min_b]

        # Warm-up: keep source discriminability first (especially important for ENTER/REFED)
        if epoch <= warm_transfer or tw <= 0.0:
            transfer_loss = torch.tensor(0.0, device=device)
            tw = 0.0
        else:
            transfer_loss = dann_loss(f_s_b, f_t_b)        # -------------------------
        # CSCFA (Eq. 6-8)
        # -------------------------
        if epoch <= warm_transfer or float(config.get('cscfa_weight', 1.0)) <= 0.0:
            cscfa_total = torch.tensor(0.0, device=device)
        else:
            src_labels_idx = torch.argmax(y_s, dim=1)
            cscfa_total, _ = cscfa_module(f_s, f_t, src_labels_idx, d_s)
            cscfa_total = float(config.get('cscfa_weight', 1.0)) * cscfa_total

        # -------------------------
        # CDAN (Eq. 10) after warm-up
        # -------------------------
        cdan_loss_val = torch.tensor(0.0, device=device)
        if cdan_enabled and model.cdan_loss_fn is not None:
            # Source uses TRUE labels (one-hot)
            y_s_cdan = y_s[perm_s][:min_b].detach()

            # Target uses high-confidence pseudo labels
            max_prob, _ = prob_t.max(dim=1)
            mask = max_prob >= float(config.get('pseudo_label_threshold', 0.95))

            # Diagnostics: count confident target samples for CDAN
            cdan_total += int(mask.numel())
            cdan_selected += int(mask.sum().item())

            if mask.sum() > 0:
                f_t_sel = f_t[mask]
                y_t_sel = prob_t[mask].detach()  # pseudo labels (probabilities)
                cdan_loss_val = model.cdan_loss_fn(f_s_b, y_s_cdan, f_t_sel, y_t_sel)
            else:
                cdan_loss_val = torch.tensor(0.0, device=device)

        # -------------------------
        # Total loss (Eq. 11)
        # -------------------------
        total_loss = pil_loss + tw * transfer_loss + cscfa_total

        if cdan_enabled:
            # NOTE: start the GRL/CDAN schedule AFTER warm-up, so λ starts near 0
            warm = int(config.get('warm_up_epochs', 0))
            denom = max(1, int(config['max_iter']) - warm)
            progress = float(max(0, epoch - warm)) / float(denom)
            progress = max(0.0, min(1.0, progress))
            lam = lambda_schedule(progress)
            total_loss = total_loss + float(config['cdan_weight']) * lam * cdan_loss_val

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Accumulate for logging
        total_loss_sum += total_loss.item()
        pil_loss_sum += pil_loss.item()
        current_trans = (tw * transfer_loss.item()) + cscfa_total.item()
        if cdan_enabled:
            current_trans += float(config['cdan_weight']) * lam * cdan_loss_val.item()
        trans_loss_sum += current_trans

    model.update_threshold(epoch)

    # Store CDAN coverage stats on the model for logging
    if cdan_enabled:
        model.cdan_total = int(cdan_total)
        model.cdan_selected = int(cdan_selected)
        model.cdan_coverage = float(cdan_selected) / float(max(1, cdan_total))
    else:
        model.cdan_total = 0
        model.cdan_selected = 0
        model.cdan_coverage = 0.0

    steps = max(1, num_batches)
    return total_loss_sum / steps, pil_loss_sum / steps, trans_loss_sum / steps


def train_subject(test_id: int,
                  config: Dict,
                  dataset_name: str = 'SEED',
                  data_dir: str = 'data',
                  mode: str = 'cross-subject',
                  session: Optional[int] = None,
                  sessions: Optional[list[int]] = None,
                  save_model: bool = True,
                  logger: 'Optional[ExperimentLogger]' = None,
                  num_subjects: Optional[int] = None) -> Tuple[float, int]:
    """Train on one target subject (LOSO).

    Args:
        test_id: 0-based subject index within the sorted subject list.
        mode: 'cross-subject' (single session) or 'cross-session' (concat sessions).
        session: required when mode='cross-subject'
        sessions: required when mode='cross-session'
    """
    setup_seed(config['seed'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_classes = config['num_classes']

    if mode == 'cross-subject':
        if session is None:
            raise ValueError("session must be provided when mode='cross-subject'")
        target_set, source_set = get_dataset(test_id, session, dataset_name, data_dir)
        split_desc = f"Session {session}"
        split_tag = f"s{session}"
    elif mode == 'cross-session':
        if not sessions:
            raise ValueError("sessions must be provided when mode='cross-session'")
        target_set, source_set = get_dataset_cross_session(test_id, sessions, dataset_name, data_dir)
        split_desc = f"Sessions {sessions} (concat)"
        split_tag = "cs" + "".join(str(s) for s in sessions)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    loader_source = Data.DataLoader(
        Data.TensorDataset(torch.from_numpy(source_set['feature']),
                           torch.from_numpy(source_set['label']),
                           torch.from_numpy(source_set['domain'])),
        batch_size=config['batch_size'], shuffle=True, drop_last=True
    )
    loader_target = Data.DataLoader(
        Data.TensorDataset(torch.from_numpy(target_set['feature']),
                           torch.from_numpy(target_set['label'])),
        batch_size=config['batch_size'], shuffle=True, drop_last=True
    )

    # Total training steps for GRL schedule (batch-level)
    steps_per_epoch = min(len(loader_source), len(loader_target))
    total_steps = config['max_iter'] * max(1, steps_per_epoch)

    model = CCAPNet(
        input_dim=310,
        hidden_dim=config['hidden_1'],
        feature_dim=config['hidden_2'],
        num_classes=num_classes,
        max_iter=config['max_iter'],
        upper_threshold=config['upper_threshold'],
        lower_threshold=config['lower_threshold']
    ).to(device)
    model.apply(weight_init)

    # DANN discriminator & loss
    domain_disc = DomainDiscriminator(config['hidden_2']).to(device)
    domain_disc.apply(weight_init)
    dann_loss = DomainAdversarialLoss(domain_disc, max_iter=total_steps).to(device)

    # CDAN discriminator & loss
    cond_disc = ConditionalDomainDiscriminator(
        in_dim=config['hidden_2'] * num_classes, hidden_dim=128
    ).to(device)
    cond_disc.apply(weight_init)
    cdan_loss_fn = ConditionalDomainAdversarialLoss(cond_disc, max_iter=total_steps).to(device)
    model.cdan_loss_fn = cdan_loss_fn

    # CSCFA module
    cscfa_module = CSCFAModule(feature_dim=config['hidden_2']).to(device)

    # Optimizer (RMSprop)
    optimizer = RMSprop(
        model.get_parameters() +
        [{'params': domain_disc.parameters()}, {'params': cond_disc.parameters()}],
        lr=config['lr'], weight_decay=config['weight_decay']
    )

    # Preload full source/target tensors for prototype update + evaluation
    source_features = torch.from_numpy(source_set['feature']).to(device)
    source_labels = torch.from_numpy(source_set['label']).to(device)
    target_features = torch.from_numpy(target_set['feature']).to(device)
    target_labels = torch.from_numpy(target_set['label']).to(device)


    # Initialize prototypes before the first training epoch (Eq. 13)
    # This avoids starting from a completely random prototype matrix.
    model.cluster_label_update(source_features, source_labels)
    best_acc = 0.0
    best_epoch = 0
    best_state = None

    total_subj = num_subjects if num_subjects is not None else "?"
    msg = f"Training subject {test_id + 1}/{total_subj}, {split_desc}, Dataset {dataset_name}"
    if logger:
        logger.log(msg)
    else:
        print(msg)

    for epoch in range(1, config['max_iter'] + 1):
        cdan_enabled = epoch > config['warm_up_epochs']

        avg_loss, avg_pil, avg_trans = train_epoch(
            model, loader_source, loader_target, dann_loss, cscfa_module,
            optimizer, epoch, config, device, cdan_enabled
        )

        # Update prototypes from all labeled source samples (Eq. 13)
        src_acc, _ = model.cluster_label_update(source_features, source_labels)

        # Evaluate on target
        tgt_acc, _ = model.evaluate(target_features, target_labels)

        if epoch % 10 == 0:
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            save_mark = " [SAVE]" if tgt_acc > best_acc else ""
            cdan_cov = getattr(model, 'cdan_coverage', 0.0) * 100.0
            log_line = (f"[{now_str}] Epoch {epoch:4d}/{config['max_iter']} | "
                        f"Loss: {avg_loss:.4f} (PIL: {avg_pil:.4f}, Trans: {avg_trans:.4f}) | "
                        f"Src Acc: {src_acc * 100:.2f}% | Tgt Acc: {tgt_acc * 100:.2f}%{save_mark} | "
                        f"CDANcov: {cdan_cov:.1f}%")
            if logger:
                logger.log(log_line)
            else:
                print(log_line)

        if tgt_acc > best_acc:
            best_acc = tgt_acc
            best_epoch = epoch
            if save_model:
                best_state = {
                    'model': model.state_dict(),
                    'epoch': epoch,
                    'accuracy': best_acc,
                    'config': config
                }

    print(f"  Best: {best_acc:.4f} @ Epoch {best_epoch}")

    if save_model and best_state is not None:
        mode_dir = mode.replace('-', '_')
        model_dir = os.path.join('results', 'models', dataset_name, mode_dir)
        os.makedirs(model_dir, exist_ok=True)
        torch.save(best_state, os.path.join(model_dir, f'best_model_{split_tag}_sub{test_id}.pth'))

    return best_acc, best_epoch


def train_session(session: int,
                  config: Dict,
                  dataset_name: str = 'SEED',
                  data_dir: str = 'data',
                  logger: 'Optional[ExperimentLogger]' = None,
                  test_id: 'Optional[int]' = None) -> np.ndarray:
    """Cross-subject training for one session (all subjects or one subject)."""
    subject_nums = list_subject_numbers(dataset_name, session, data_dir)
    num_subjects = len(subject_nums)
    best_acc_mat = np.zeros(num_subjects)

    if logger:
        logger.log(f"\n{'=' * 70}")
        logger.log(f"{dataset_name} Session {session} Training Start")
        logger.log(f"{'=' * 70}")

    subjects = range(num_subjects) if test_id is None else [test_id]
    if test_id is not None:
        print(f"Debug: only training subject {test_id}")

    for subject_id in subjects:
        best_acc, _ = train_subject(
            subject_id,
            config,
            dataset_name=dataset_name,
            data_dir=data_dir,
            mode='cross-subject',
            session=session,
            sessions=None,
            logger=logger,
            num_subjects=num_subjects,
        )
        best_acc_mat[subject_id] = best_acc

    valid_accs = best_acc_mat[best_acc_mat > 0]
    if len(valid_accs) > 0:
        msg = f"\nSession {session} Finished. Best Average Target Accuracy: {np.mean(valid_accs) * 100:.2f}%"
        if len(valid_accs) > 1:
            msg += f" ± {np.std(valid_accs) * 100:.2f}%"
    else:
        msg = f"\nSession {session} Finished. No subjects trained."

    if logger:
        logger.log(msg)
    else:
        print(msg)

    return best_acc_mat


def train_cross_session(sessions: list[int],
                        config: Dict,
                        dataset_name: str = 'SEED',
                        data_dir: str = 'data',
                        logger: 'Optional[ExperimentLogger]' = None,
                        test_id: 'Optional[int]' = None) -> np.ndarray:
    """Cross-session training: concatenate sessions per subject, then LOSO."""
    common_subject_nums = list_common_subject_numbers(dataset_name, sessions, data_dir)
    num_subjects = len(common_subject_nums)
    best_acc_mat = np.zeros(num_subjects)

    if logger:
        logger.log(f"\n{'=' * 70}")
        logger.log(f"{dataset_name} Cross-Session Training Start | sessions={sessions}")
        logger.log(f"{'=' * 70}")

    subjects = range(num_subjects) if test_id is None else [test_id]
    if test_id is not None:
        print(f"Debug: only training subject {test_id}")

    for subject_id in subjects:
        best_acc, _ = train_subject(
            subject_id,
            config,
            dataset_name=dataset_name,
            data_dir=data_dir,
            mode='cross-session',
            session=None,
            sessions=sessions,
            logger=logger,
            num_subjects=num_subjects,
        )
        best_acc_mat[subject_id] = best_acc

    valid_accs = best_acc_mat[best_acc_mat > 0]
    if len(valid_accs) > 0:
        msg = (f"\nCross-Session Finished. Best Average Target Accuracy: "
               f"{np.mean(valid_accs) * 100:.2f}%")
        if len(valid_accs) > 1:
            msg += f" ± {np.std(valid_accs) * 100:.2f}%"
    else:
        msg = "\nCross-Session Finished. No subjects trained."

    if logger:
        logger.log(msg)
    else:
        print(msg)

    return best_acc_mat


def main():
    parser = argparse.ArgumentParser(description='CCAP-Net Training')
    parser.add_argument('--dataset', type=str, default='SEED',
                        choices=['SEED', 'SEED-IV', 'ENTER', 'REFED', 'ALL'])
    parser.add_argument('--mode', type=str, default='cross-subject',
                        choices=['cross-subject', 'cross-session'],
                        help='Experiment protocol: cross-subject or cross-session')
    parser.add_argument('--sessions', type=int, nargs='+', default=[1, 2, 3],
                        help='Sessions to run (cross-subject) or to concatenate (cross-session).')
    parser.add_argument('--max_iter', type=int, default=1200)
    parser.add_argument('--test_id', type=int, default=None,
                        help='Target subject index (0-based). Range depends on dataset.')
    parser.add_argument('--transfer_weight', type=float, default=0.5, help='DANN weight (0.0 to disable DANN)')
    parser.add_argument('--pseudo_label_threshold', type=float, default=None,
                        help='CDAN pseudo-label confidence threshold. If not set, use dataset default.')
    parser.add_argument('--warm_up_epochs', type=int, default=None,
                        help='Warm-up epochs before enabling CDAN. If not set, use dataset default.')
    parser.add_argument('--warm_up_transfer_epochs', type=int, default=None,
                        help='Warm-up epochs before enabling DANN/CSCFA. If not set, use dataset default.')
    parser.add_argument('--seed', type=int, default=20)
    parser.add_argument('--data_dir', type=str, default='data')
    args = parser.parse_args()

    # Expand dataset list
    dataset_list = ['SEED', 'SEED-IV', 'ENTER', 'REFED'] if args.dataset == 'ALL' else [args.dataset]

    for dataset_name in dataset_list:
        spec = get_dataset_info(dataset_name)
        num_classes = int(spec['num_classes'])
        valid_sessions = list(spec['sessions'])

        # Filter user-provided sessions to what's available for the dataset
        sessions = [s for s in args.sessions if s in valid_sessions]
        if len(sessions) == 0:
            # If user didn't include any valid session (e.g., ENTER but sessions=[1,2,3]), fall back.
            sessions = valid_sessions
        # Dataset-specific defaults (can be overridden via CLI args)
        dataset_hparam_defaults = {
            'SEED': {'pseudo_label_threshold': 0.95, 'warm_up_epochs': 100, 'warm_up_transfer_epochs': 0},
            'SEED-IV': {'pseudo_label_threshold': 0.95, 'warm_up_epochs': 100, 'warm_up_transfer_epochs': 0},
            'ENTER': {'pseudo_label_threshold': 0.60, 'warm_up_epochs': 200, 'warm_up_transfer_epochs': 100},
            'REFED': {'pseudo_label_threshold': 0.60, 'warm_up_epochs': 200, 'warm_up_transfer_epochs': 100},
        }
        ds_defaults = dataset_hparam_defaults.get(dataset_name, {})

        pseudo_label_threshold = (
            float(args.pseudo_label_threshold)
            if args.pseudo_label_threshold is not None
            else float(ds_defaults.get('pseudo_label_threshold', 0.95))
        )
        warm_up_epochs = (
            int(args.warm_up_epochs)
            if args.warm_up_epochs is not None
            else int(ds_defaults.get('warm_up_epochs', 100))
        )
        # Safety clamp
        warm_up_epochs = max(0, warm_up_epochs)
        if warm_up_epochs >= int(args.max_iter):
            warm_up_epochs = max(0, int(args.max_iter) - 1)

        warm_up_transfer_epochs = (
            int(args.warm_up_transfer_epochs)
            if args.warm_up_transfer_epochs is not None
            else int(ds_defaults.get('warm_up_transfer_epochs', 0))
        )
        warm_up_transfer_epochs = max(0, warm_up_transfer_epochs)
        if warm_up_transfer_epochs >= int(args.max_iter):
            warm_up_transfer_epochs = max(0, int(args.max_iter) - 1)

        # Build config (paper's Table-1 settings; only num_classes differs)
        config = {
            'hidden_1': 128,
            'hidden_2': 64,
            'num_classes': num_classes,
            'batch_size': 96,
            'max_iter': args.max_iter,
            'warm_up_epochs': warm_up_epochs,
            'warm_up_transfer_epochs': warm_up_transfer_epochs,
            'lr': 2.356e-3,
            'weight_decay': 1e-5,
            'upper_threshold': 0.9,
            'lower_threshold': 0.5,
            'cdan_weight': 1.0,
            'cscfa_weight': 1.0,
            'ce_weight': 1.0,
            'pseudo_label_threshold': pseudo_label_threshold,
            'transfer_weight': args.transfer_weight,
            'seed': args.seed,
        }


        # Log directory
        log_dir = os.path.join('results', 'logs')
        os.makedirs(log_dir, exist_ok=True)

        # Hyperparameter config file (include dataset+mode to avoid overwrite)
        mode_dir = args.mode.replace('-', '_')
        config_path = os.path.join(log_dir, f'hyperparameter_config_{dataset_name}_{mode_dir}.txt')
        save_hyperparameter_config(config, config_path)
        print(f"Hyperparameter config saved to: {config_path}")

        # Logger
        log_filename = f"train_{dataset_name}_{mode_dir}.log"
        log_path = os.path.join(log_dir, log_filename)
        logger = ExperimentLogger(log_path)

        logger.log("=" * 70)
        logger.log(f"Experiment Log: CCAP-Net | dataset={dataset_name} | mode={args.mode} | sessions={sessions}")
        logger.log(f"Start Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.log("=" * 70)
        logger.log("")

        all_acc: list[float] = []

        if args.mode == 'cross-subject':
            for session in sessions:
                best_acc_mat = train_session(
                    session,
                    config,
                    dataset_name,
                    args.data_dir,
                    logger=logger,
                    test_id=args.test_id,
                )
                all_acc.extend(best_acc_mat.tolist())

                # Save results
                result_dict = {'best_acc_mat': best_acc_mat, 'config': config}
                out_dir = os.path.join('results', dataset_name, mode_dir)
                os.makedirs(out_dir, exist_ok=True)
                scio.savemat(os.path.join(out_dir, f'result_session{session}.mat'), result_dict)

        else:  # cross-session
            best_acc_mat = train_cross_session(
                sessions=sessions,
                config=config,
                dataset_name=dataset_name,
                data_dir=args.data_dir,
                logger=logger,
                test_id=args.test_id,
            )
            all_acc.extend(best_acc_mat.tolist())

            result_dict = {'best_acc_mat': best_acc_mat, 'config': config, 'sessions': sessions}
            out_dir = os.path.join('results', dataset_name, mode_dir)
            os.makedirs(out_dir, exist_ok=True)
            scio.savemat(os.path.join(out_dir, f'result_cross_session.mat'), result_dict)

        all_acc_arr = np.array([x for x in all_acc if x > 0], dtype=np.float32)
        if all_acc_arr.size > 0:
            logger.log(f"\n{'=' * 70}")
            logger.log(f"Overall Average: {all_acc_arr.mean() * 100:.2f}% ± {all_acc_arr.std() * 100:.2f}%")
            logger.log(f"{'=' * 70}")
        else:
            logger.log(f"\n{'=' * 70}")
            logger.log("Overall Average: (no valid runs)")
            logger.log(f"{'=' * 70}")

        logger.close()
        print(f"\nTraining finished for dataset={dataset_name}! Logs are in results/logs/.")


if __name__ == "__main__":
    main()
