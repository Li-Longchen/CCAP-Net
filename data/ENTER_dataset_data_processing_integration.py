import numpy as np
import scipy.io as sio
import scipy.signal as signal
import os
import h5py

# =====================
#  Filter helpers
# =====================

def butter_bandpass(lowcut, highcut, fs, order=5):
    """Return second‑order‑sections (SOS) band‑pass filter."""
    return signal.butter(order, [lowcut, highcut], btype="band", fs=fs, output="sos")


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Zero‑phase band‑pass filtering (channel × time -> same shape)."""
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    return signal.sosfiltfilt(sos, data, axis=1)  # zero‑phase, no delay


# =====================
#  Feature helpers
# =====================

def calculate_de(data):
    """Differential entropy per‑channel with numeric stability."""
    variance = np.var(data, axis=1, ddof=1)
    eps = max(np.finfo(data.dtype).eps, 1e-8 * variance.mean())
    return 0.5 * np.log2(2 * np.pi * np.e * (variance + eps))


# =====================
#  Down‑sampling helper
# =====================

def downsample(data, orig_fs, target_fs):
    """Linear‑phase FIR decimation, preserves γ‑band up to 50 Hz."""
    factor = orig_fs // target_fs
    return signal.decimate(data, factor, axis=1, ftype="fir")


# =====================
#  Main per‑trial processing
# =====================

def process_and_extract_features(data, orig_fs=1000, target_fs=200):
    print(f"Original data shape: {data.shape}")

    # shape check (channels, time)
    if data.shape[0] != 62:
        data = data.T
        print(f"Transposed data shape: {data.shape}")

    # 至少要有 2 s 基线 + 4 s 激活
    if data.shape[1] < orig_fs * 6:
        raise ValueError("数据长度不足，无法处理 (需要≥6 s)")

    # 下采样 (1000 Hz → 200 Hz)
    data_ds = downsample(data, orig_fs, target_fs)
    print(f"Downsampled data shape: {data_ds.shape}")

    # =====  Five canonical bands  =====
    data_delta = butter_bandpass_filter(data_ds, 1, 4, target_fs, order=3)
    data_theta = butter_bandpass_filter(data_ds, 4, 8, target_fs, order=3)
    data_alpha = butter_bandpass_filter(data_ds, 8, 14, target_fs, order=3)
    data_beta  = butter_bandpass_filter(data_ds, 14, 31, target_fs, order=3)
    data_gamma = butter_bandpass_filter(data_ds, 31, 50, target_fs, order=3)

    # =====  Skip first 2 s as baseline (no DE subtraction)  =====
    baseline_pts = 2 * target_fs

    window_size = 4 * target_fs  # 4 s window
    step_size   = window_size    # non‑overlapping

    n_samples = int((data_ds.shape[1] - baseline_pts - window_size) // step_size + 1)
    print(f"Number of samples: {n_samples}")

    de_features = []
    for i in range(n_samples):
        start = baseline_pts + i * step_size
        end   = start + window_size

        seg_delta = data_delta[:, start:end]
        seg_theta = data_theta[:, start:end]
        seg_alpha = data_alpha[:, start:end]
        seg_beta  = data_beta[:,  start:end]
        seg_gamma = data_gamma[:, start:end]

        de_delta = calculate_de(seg_delta)
        de_theta = calculate_de(seg_theta)
        de_alpha = calculate_de(seg_alpha)
        de_beta  = calculate_de(seg_beta)
        de_gamma = calculate_de(seg_gamma)

        de_feat = np.concatenate([de_delta, de_theta, de_alpha, de_beta, de_gamma])  # 310‑d
        de_features.append(de_feat)

    de_features = np.array(de_features)
    print(f"de_features shape: {de_features.shape}")

    if de_features.shape[0] == 0:
        raise ValueError("数据不足以进行特征计算")

    # reshape → (num_samples, 62, 5)
    de_activated = de_features.reshape(de_features.shape[0], 62, 5)
    return de_activated


# =====================
#  I/O – unchanged except shape comment
# =====================

def read_mat_file(file_path):
    try:
        mat_data = sio.loadmat(file_path)
        return mat_data["eegdata"]
    except NotImplementedError:
        with h5py.File(file_path, "r") as f:
            return f["eegdata"][:]


def main(root_path, orig_fs=1000, target_fs=200):
    save_root = os.path.join(root_path, "TYUT3.0_feature", "session1")
    if not os.path.isdir(save_root):
        os.makedirs(save_root, exist_ok=True)
    for peo in range(1, 51):
        dataset_dir = os.path.join(root_path, str(peo), "EEG")
        all_features = []
        all_labels   = []
        for video in range(1, 61):
            file_path = os.path.join(dataset_dir, f"EEG_{video}.mat")
            try:
                eeg_data = read_mat_file(file_path)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue

            try:
                features = process_and_extract_features(eeg_data, orig_fs, target_fs)
            except ValueError as e:
                print(f"Skipping {file_path} due to error: {e}")
                continue

            if features.size == 0:
                print(f"No features extracted from {file_path}")
                continue

            # 标签映射逻辑保持不变
            if video <= 15:
                label = 0
            elif video <= 30:
                label = 1
            elif video <= 45:
                label = 2
            else:
                label = 3

            labels = np.full(features.shape[0], label)

            all_features.append(features)
            all_labels.append(labels)

        # ---------- 保存每个被试 ----------
        if len(all_features) == 0 or len(all_labels) == 0:
            print(f"No valid features or labels for subject {peo}")
            continue

        # 1) 拼接
        all_features = np.concatenate(all_features)  # (N, 62, 5)
        all_labels = np.concatenate(all_labels)  # (N,)

        # 2) reshape → (N, 310)  &  (N, 1)
        feature_2d = all_features.reshape(all_features.shape[0], -1)  # (N, 310)
        label_col = all_labels.reshape(-1, 1)  # (N, 1)

        # 3) 构造数据结构体（与 MATLAB struct 对齐）
        dataset_session1 = {
            "feature": feature_2d,
            "label": label_col
        }

        # 4) 保存路径与文件名
        save_path = os.path.join(save_root,f"sub_{peo}_session_1.mat")
        sio.savemat(save_path, {"dataset_session1": dataset_session1})

        print(f"Saved subject {peo}: feature {feature_2d.shape}, label {label_col.shape}")


# =====================
#  Run the pipeline
# =====================
if __name__ == "__main__":
    root_path = "D:\\TYUT3.0"
    main(root_path)
    print("所有特征和标签已成功保存。")
