# -*- coding: utf-8 -*-
"""utils.py

Utility functions for CCAP-Net.

This file is responsible for loading the pre-processed DE feature .mat files
and constructing Leave-One-Subject-Out (LOSO) splits.

Paper-aligned (multi-source) setting:
  - Each non-target subject is treated as an independent source domain.
  - Therefore, `source_set` additionally provides a per-sample `domain` vector
    (subject id) for CSCFA.

Supported datasets:
  - SEED      (3 classes, sessions 1/2/3)
  - SEED-IV   (4 classes, sessions 1/2/3)
  - ENTER     (4 classes, session 1)
  - REFED     (5 classes, session 1)

All datasets are expected to be saved as MATLAB structs:
  dataset_session{K}.feature : (N, 310)
  dataset_session{K}.label   : (N, 1) or (N,)
with filenames like: sub_{id}_session_{K}.mat

Notes on directory layout:
  The training scripts default to `data_dir=<project>/data` and look for:
    data_dir/<DATASET_NAME>/sessionK/*.mat

"""

from __future__ import annotations

import os
import re
from typing import Dict, Tuple, List, Optional, Iterable, Set

import numpy as np
import scipy.io as scio
from sklearn import preprocessing


# -----------------------------------------------------------------------------
# Dataset specs
# -----------------------------------------------------------------------------

DATASET_SPECS: Dict[str, Dict] = {
    # Standard layout expected by the repo: data_dir/SEED/session{1,2,3}/
    "SEED": {
        "num_classes": 3,
        "sessions": [1, 2, 3],
        "dir_candidates": [
            "SEED",           # recommended
            "feature",        # from SEED processing script
            "SEED_feature",   # common variant
        ],
    },
    # Standard layout: data_dir/SEED-IV/session{1,2,3}/
    "SEED-IV": {
        "num_classes": 4,
        "sessions": [1, 2, 3],
        "dir_candidates": [
            "SEED-IV",        # recommended
            "SEEDIV_feature", # from SEED-IV processing script
            "SEED-IV_feature",# common variant
        ],
    },
    # ENTER (script saves into TYUT3.0_feature/session1)
    "ENTER": {
        "num_classes": 4,
        "sessions": [1],
        "dir_candidates": [
            "ENTER",          # recommended
            "ENTER_feature",  # common variant
            "TYUT3.0_feature",# from ENTER processing script
        ],
    },
    # REFED (script saves into REFED_feature/ (no session folder))
    "REFED": {
        "num_classes": 5,
        "sessions": [1],
        "dir_candidates": [
            "REFED",          # recommended
            "REFED_feature",  # from REFED processing script
        ],
    },
}


def get_dataset_info(dataset_name: str) -> Dict:
    """Return dataset metadata (num_classes, available sessions, etc.)."""
    if dataset_name not in DATASET_SPECS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: {sorted(DATASET_SPECS.keys())}")
    return dict(DATASET_SPECS[dataset_name])


# -----------------------------------------------------------------------------
# Path + filename helpers
# -----------------------------------------------------------------------------


def _parse_subject_number(file_name: str) -> Optional[int]:
    """Extract subject number from a .mat filename.

    Expected patterns:
      - sub_{id}_session_{k}.mat  -> id
      - ..._0.mat / ..._14.mat    -> last integer before .mat
    """
    m = re.search(r"sub_(\d+)", file_name)
    if m:
        return int(m.group(1))

    # Fallback: last integer before extension
    m = re.search(r"(\d+)(?=\.mat$)", file_name)
    if m:
        return int(m.group(1))
    return None


def _candidate_session_dirs(dataset_name: str, session: int, data_dir: str) -> List[str]:
    spec = get_dataset_info(dataset_name)
    candidates: List[str] = []

    # Try dataset-specific candidates
    for base in spec.get("dir_candidates", []):
        # Most common: <data_dir>/<base>/sessionK
        candidates.append(os.path.join(data_dir, base, f"session{session}"))

        # Some datasets (e.g., REFED_feature) store mats directly without session folder
        candidates.append(os.path.join(data_dir, base))

    # Also allow passing a directory that is already at dataset root or session root
    candidates.append(os.path.join(data_dir, f"session{session}"))
    candidates.append(data_dir)

    # De-duplicate while preserving order
    seen: Set[str] = set()
    unique: List[str] = []
    for p in candidates:
        p_norm = os.path.normpath(p)
        if p_norm not in seen:
            unique.append(p_norm)
            seen.add(p_norm)
    return unique


def _resolve_session_dir(dataset_name: str, session: int, data_dir: str) -> str:
    """Resolve a session directory that contains the .mat files."""
    tried = _candidate_session_dirs(dataset_name, session, data_dir)
    for p in tried:
        if not os.path.isdir(p):
            continue
        mats = [f for f in os.listdir(p) if f.endswith(".mat")]
        if len(mats) > 0:
            return p

    msg = (
        f"Could not find .mat files for dataset={dataset_name}, session={session}.\n"
        f"data_dir={os.path.abspath(data_dir)}\n"
        f"Tried the following directories:\n  - " + "\n  - ".join(tried)
    )
    raise FileNotFoundError(msg)


def list_subject_numbers(dataset_name: str, session: int, data_dir: str) -> List[int]:
    """List subject numbers (as they appear in filenames) for a given session."""
    session_dir = _resolve_session_dir(dataset_name, session, data_dir)
    mat_files = [f for f in os.listdir(session_dir) if f.endswith(".mat")]

    subj_nums: List[int] = []
    for f in mat_files:
        sid = _parse_subject_number(f)
        if sid is not None:
            subj_nums.append(sid)

    subj_nums = sorted(set(subj_nums))
    if len(subj_nums) == 0:
        raise FileNotFoundError(
            f"No valid subject .mat files found in {session_dir}. "
            f"Expected filenames like sub_{{id}}_session_{{k}}.mat"
        )
    return subj_nums


def list_common_subject_numbers(dataset_name: str, sessions: Iterable[int], data_dir: str) -> List[int]:
    """Return the intersection of subject numbers across multiple sessions."""
    sessions = list(sessions)
    if len(sessions) == 0:
        raise ValueError("sessions must be a non-empty list")

    common: Optional[Set[int]] = None
    for s in sessions:
        s_ids = set(list_subject_numbers(dataset_name, s, data_dir))
        common = s_ids if common is None else (common & s_ids)

    out = sorted(common) if common is not None else []
    if len(out) == 0:
        raise FileNotFoundError(
            f"No common subjects found across sessions={sessions} for dataset={dataset_name}."
        )
    return out


# -----------------------------------------------------------------------------
# Loading + preprocessing helpers
# -----------------------------------------------------------------------------


def _normalize_label_values(labels: np.ndarray, dataset_name: str, num_classes: int) -> np.ndarray:
    """Normalize raw label values into [0, num_classes-1]."""
    lab = np.asarray(labels).reshape(-1).astype(np.int64)

    # Common special case: SEED labels might be {-1,0,1} (neg/neu/pos)
    if dataset_name == "SEED":
        uniq = set(np.unique(lab).tolist())
        if uniq.issubset({-1, 0, 1}):
            lab = lab + 1
        elif uniq.issubset({1, 2, 3}):
            lab = lab - 1

    # Generic case: labels are 1..C
    if lab.size > 0 and lab.min() == 1 and lab.max() == num_classes:
        lab = lab - 1

    if lab.size > 0 and (lab.min() < 0 or lab.max() >= num_classes):
        raise ValueError(
            f"Label values out of range after normalization for dataset={dataset_name}. "
            f"min={lab.min()}, max={lab.max()}, num_classes={num_classes}."
        )
    return lab


def _to_one_hot(label_int: np.ndarray, num_classes: int) -> np.ndarray:
    label_int = np.asarray(label_int, dtype=np.int64).reshape(-1)
    one_hot = np.eye(num_classes, dtype=np.float32)[label_int]
    return one_hot


def _scale_feature_minmax(feature: np.ndarray) -> np.ndarray:
    """Per-subject MinMax scaling to [-1, 1] (keeps behavior of original code)."""
    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    return scaler.fit_transform(feature).astype(np.float32)


def _load_subject_mat(full_path: str, mat_key: str) -> Tuple[np.ndarray, np.ndarray]:
    mat_dict = scio.loadmat(full_path)
    if mat_key not in mat_dict:
        # Some datasets only have dataset_session1; if session=1, allow fallback.
        if mat_key != "dataset_session1" and "dataset_session1" in mat_dict:
            mat_key = "dataset_session1"
        else:
            avail = [k for k in mat_dict.keys() if k.startswith("dataset_session")]
            raise KeyError(f"Key '{mat_key}' not found in {full_path}. Available: {avail}")

    feature = mat_dict[mat_key]["feature"][0, 0]
    label = mat_dict[mat_key]["label"][0, 0]
    return feature, label


def _load_all_subjects_one_session(dataset_name: str,
                                  session: int,
                                  data_dir: str,
                                  scale: bool = True) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Load all subjects for a single session into ordered lists."""
    spec = get_dataset_info(dataset_name)
    num_classes = int(spec["num_classes"])

    session_dir = _resolve_session_dir(dataset_name, session, data_dir)
    mat_files = [f for f in os.listdir(session_dir) if f.endswith(".mat")]

    subj_to_file: Dict[int, str] = {}
    for f in mat_files:
        sid = _parse_subject_number(f)
        if sid is None:
            continue
        subj_to_file[sid] = os.path.join(session_dir, f)

    if len(subj_to_file) == 0:
        raise FileNotFoundError(
            f"No subject .mat files found in {session_dir}. "
            f"Expected filenames like sub_{{id}}_session_{{k}}.mat"
        )

    subject_nums = sorted(subj_to_file.keys())

    feature_list: List[np.ndarray] = []
    label_list: List[np.ndarray] = []
    mat_key = f"dataset_session{session}"

    for sid in subject_nums:
        feat, lab = _load_subject_mat(subj_to_file[sid], mat_key)
        if scale:
            feat = _scale_feature_minmax(feat)
        lab_int = _normalize_label_values(lab, dataset_name, num_classes)
        one_hot = _to_one_hot(lab_int, num_classes)
        feature_list.append(feat.astype(np.float32))
        label_list.append(one_hot.astype(np.float32))

    return feature_list, label_list


# -----------------------------------------------------------------------------
# Public APIs used by training/evaluation
# -----------------------------------------------------------------------------


def get_dataset(test_id: int,
                session: int,
                dataset_name: str = "SEED",
                data_dir: str = "data") -> Tuple[Dict, Dict]:
    """Load dataset and build LOSO split (single session / cross-subject).

    Args:
        test_id: target subject index (0-based within the sorted subject list)
        session: session number
        dataset_name: one of {'SEED','SEED-IV','ENTER','REFED'}
        data_dir: root directory

    Returns:
        target_set: {'feature': (N_t,310), 'label': (N_t,C)}
        source_set: {'feature': (N_s,310), 'label': (N_s,C), 'domain': (N_s,)}
    """
    feature_list, label_list = _load_all_subjects_one_session(dataset_name, session, data_dir, scale=True)

    if test_id < 0 or test_id >= len(feature_list):
        raise ValueError(f"test_id={test_id} out of range, total subjects={len(feature_list)}")

    target_feature = feature_list[test_id]
    target_label = label_list[test_id]

    source_features: List[np.ndarray] = []
    source_labels: List[np.ndarray] = []
    source_domains: List[np.ndarray] = []

    for subj_idx in range(len(feature_list)):
        if subj_idx == test_id:
            continue
        f = feature_list[subj_idx]
        l = label_list[subj_idx]
        d = np.full((f.shape[0],), subj_idx, dtype=np.int64)

        source_features.append(f)
        source_labels.append(l)
        source_domains.append(d)

    source_feature = np.vstack(source_features).astype(np.float32)
    source_label = np.vstack(source_labels).astype(np.float32)
    source_domain = np.concatenate(source_domains).astype(np.int64)

    target_set = {"feature": target_feature.astype(np.float32), "label": target_label.astype(np.float32)}
    source_set = {"feature": source_feature, "label": source_label, "domain": source_domain}
    return target_set, source_set


def get_dataset_cross_session(test_id: int,
                              sessions: List[int],
                              dataset_name: str = "SEED",
                              data_dir: str = "data") -> Tuple[Dict, Dict]:
    """LOSO split for cross-session setting.

    For each subject:
      - concatenate all specified sessions into one subject-specific set
    Then:
      - choose one subject as target (test_id)
      - all remaining subjects are sources (multi-source; domain id = subject index)

    Notes:
      - For datasets with only one session (ENTER/REFED), this degenerates to
        the same split as get_dataset(..., session=1, ...).
    """
    sessions = list(sessions)
    if len(sessions) == 0:
        raise ValueError("sessions must be a non-empty list")

    spec = get_dataset_info(dataset_name)
    num_classes = int(spec["num_classes"])

    common_subject_nums = list_common_subject_numbers(dataset_name, sessions, data_dir)

    if test_id < 0 or test_id >= len(common_subject_nums):
        raise ValueError(f"test_id={test_id} out of range, total subjects={len(common_subject_nums)}")

    # Load per-session data into dicts keyed by subject number.
    per_session_data: List[Tuple[int, Dict[int, np.ndarray], Dict[int, np.ndarray]]] = []
    for s in sessions:
        session_dir = _resolve_session_dir(dataset_name, s, data_dir)
        mat_files = [f for f in os.listdir(session_dir) if f.endswith(".mat")]
        subj_to_file: Dict[int, str] = {}
        for f in mat_files:
            sid = _parse_subject_number(f)
            if sid is None:
                continue
            if sid in common_subject_nums:
                subj_to_file[sid] = os.path.join(session_dir, f)

        feat_dict: Dict[int, np.ndarray] = {}
        lab_dict: Dict[int, np.ndarray] = {}
        mat_key = f"dataset_session{s}"
        for sid in common_subject_nums:
            full_path = subj_to_file.get(sid)
            if full_path is None:
                # Should not happen because we intersected subjects, but keep safe.
                raise FileNotFoundError(f"Missing subject {sid} for session {s} in {session_dir}")
            feat, lab = _load_subject_mat(full_path, mat_key)
            feat_dict[sid] = feat
            lab_dict[sid] = lab
        per_session_data.append((s, feat_dict, lab_dict))

    # Build concatenated per-subject arrays (scale AFTER concatenation)
    subject_features: List[np.ndarray] = []
    subject_labels: List[np.ndarray] = []

    for sid in common_subject_nums:
        feats = [feat_dict[sid] for (_, feat_dict, _) in per_session_data]
        labs = [lab_dict[sid] for (_, _, lab_dict) in per_session_data]

        feat_cat = np.vstack(feats)
        lab_cat = np.vstack([np.asarray(x).reshape(-1, 1) for x in labs])

        feat_scaled = _scale_feature_minmax(feat_cat)
        lab_int = _normalize_label_values(lab_cat, dataset_name, num_classes)
        one_hot = _to_one_hot(lab_int, num_classes)

        subject_features.append(feat_scaled.astype(np.float32))
        subject_labels.append(one_hot.astype(np.float32))

    # LOSO split
    target_feature = subject_features[test_id]
    target_label = subject_labels[test_id]

    source_features: List[np.ndarray] = []
    source_labels: List[np.ndarray] = []
    source_domains: List[np.ndarray] = []

    for subj_idx in range(len(subject_features)):
        if subj_idx == test_id:
            continue
        f = subject_features[subj_idx]
        l = subject_labels[subj_idx]
        d = np.full((f.shape[0],), subj_idx, dtype=np.int64)
        source_features.append(f)
        source_labels.append(l)
        source_domains.append(d)

    source_feature = np.vstack(source_features).astype(np.float32)
    source_label = np.vstack(source_labels).astype(np.float32)
    source_domain = np.concatenate(source_domains).astype(np.int64)

    target_set = {"feature": target_feature.astype(np.float32), "label": target_label.astype(np.float32)}
    source_set = {"feature": source_feature, "label": source_label, "domain": source_domain}
    return target_set, source_set
