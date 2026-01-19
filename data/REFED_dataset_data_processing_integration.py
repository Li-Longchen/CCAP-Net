import os
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt

# ===========================
# 配置区
# ===========================
BASE_DIR = r"D:\REFED数据集\REFED-dataset"
DATA_DIR = os.path.join(BASE_DIR, "data")
ANNOT_DIR = os.path.join(BASE_DIR, "annotations")
OUT_DIR = os.path.join(BASE_DIR, "REFED_feature")

SRATE = 1000          # EEG采样率 1000 Hz
N_CHANNELS_KEEP = 62  # 删除最后2个通道
N_SUBJECTS = 32
N_VIDEOS = 15

# 5个频段 (low, high)
BANDS = [
    (1, 4),    # delta
    (4, 8),    # theta
    (8, 14),   # alpha
    (14, 31),  # beta
    (31, 50)   # gamma
]


def ensure_out_dir():
    """创建输出目录"""
    os.makedirs(OUT_DIR, exist_ok=True)


def val_aro_to_label(val, aro):
    """
    根据效价(valence) 和 唤醒(arousal) 映射到 0~4 标签.
        - 中效价 中唤醒 -> 0
        - 低效价 低唤醒 -> 1
        - 高效价 低唤醒 -> 2
        - 低效价 高唤醒 -> 3
        - 高效价 高唤醒 -> 4

    val >128 为高效价, <128 为低效价, ==128 为中效价
    aro >128 为高唤醒, <128 为低唤醒, ==128 为中唤醒

    未在上述5类中的情况，统一暂归为0（中性类）。
    """
    def level(x):
        if x > 128:
            return 1   # 高
        elif x < 128:
            return -1  # 低
        else:
            return 0   # 中

    v = level(val)
    a = level(aro)

    if v == 0 and a == 0:
        return 0  # 中效价中唤醒
    if v == -1 and a == -1:
        return 1  # 低效价低唤醒
    if v == 1 and a == -1:
        return 2  # 高效价低唤醒
    if v == -1 and a == 1:
        return 3  # 低效价高唤醒
    if v == 1 and a == 1:
        return 4  # 高效价高唤醒

    # 其他组合，统一归为0
    return 0


def unify_label_shape(label_array):
    """
    将标签变量统一成 (秒数, 2) 的形状：
    - 如果是 (秒数, 2)，直接返回
    - 如果是 (2, 秒数)，转置
    """
    arr = np.array(label_array)

    if arr.ndim != 2:
        raise ValueError(f"标签数组维度异常: {arr.shape}")

    if arr.shape[1] == 2:
        # (秒数, 2)
        return arr
    elif arr.shape[0] == 2:
        # (2, 秒数) -> (秒数, 2)
        return arr.T
    else:
        raise ValueError(f"无法识别的标签形状: {arr.shape}，期望为 (T,2) 或 (2,T)")


def bandpass_filter(data, low, high, fs, order=4):
    """
    对 data 进行带通滤波
    data: (channels, time)
    low, high: 频率 (Hz)
    fs: 采样率
    返回同形状数组
    """
    nyq = 0.5 * fs
    low_norm = low / nyq
    high_norm = high / nyq
    b, a = butter(order, [low_norm, high_norm], btype='bandpass')
    # 零相位滤波，避免相位畸变
    filtered = filtfilt(b, a, data, axis=-1)
    return filtered


def compute_de_features(eeg_band, n_seconds, fs=SRATE, eps=1e-10):
    """
    对某一个频段滤波后的EEG信号，计算每秒的DE特征。

    eeg_band: (channels, time) 已带通到某频段
    n_seconds: 有效秒数
    返回: feats, 形状 (n_seconds, channels)
           feats[t, ch] = 第 t 秒、第 ch 通道的 DE
    """
    n_channels, n_points = eeg_band.shape
    expected_points = n_seconds * fs
    eeg_band = eeg_band[:, :expected_points]

    feats = np.zeros((n_seconds, n_channels), dtype=np.float32)
    const_term = 0.5 * np.log(2 * np.pi * np.e)

    for sec in range(n_seconds):
        start = sec * fs
        end = (sec + 1) * fs
        segment = eeg_band[:, start:end]  # (channels, fs)
        # 方差估计
        var = np.var(segment, axis=1, ddof=1) + eps  # (channels,)
        # DE = 0.5 * ln(2*pi*e*var)
        de = const_term + 0.5 * np.log(var)
        feats[sec, :] = de.astype(np.float32)

    return feats  # (n_seconds, channels)


def process_subject(sub_id):
    """
    处理单个被试：
      1. 读取 EEG_videos.mat 和 对应的 _label.mat
      2. 对15个video逐个：
         - 删除EEG最后两个通道 -> 62 通道
         - 校正EEG秒数和标签秒数，按最小值对齐
         - 对整个trial的62通道EEG，分别做5个频段的带通滤波
         - 对每个频段、每秒、每通道计算DE
         - 把同一秒的5个频段DE组合成 (62,5) 的特征矩阵
      3. 将所有trial的特征按时间顺序拼接
      4. 展平为 (样本数, 310)，保存到 dataset_session1.feature
         标签保存到 dataset_session1.label
         文件名: sub_{id}_session_1.mat
    """
    subj_str = str(sub_id)
    eeg_file = os.path.join(DATA_DIR, subj_str, "EEG_videos.mat")
    label_file = os.path.join(ANNOT_DIR, f"{subj_str}_label.mat")

    if not os.path.exists(eeg_file):
        raise FileNotFoundError(f"找不到EEG文件: {eeg_file}")
    if not os.path.exists(label_file):
        raise FileNotFoundError(f"找不到标签文件: {label_file}")

    print(f"=== 处理被试 {sub_id} ===")
    eeg_mat = sio.loadmat(eeg_file)
    label_mat = sio.loadmat(label_file)

    feature_list = []  # 每个元素形状为 (62, 5)
    labels_list = []   # 每个元素为对应标签int

    for vid in range(1, N_VIDEOS + 1):
        key = f"video_{vid}"
        if key not in eeg_mat or key not in label_mat:
            print(f"  警告: {key} 在EEG或标签文件中不存在，跳过该trial")
            continue

        # ---------- EEG 部分 ----------
        eeg = np.array(eeg_mat[key])  # 期望形状为 (64, T)
        if eeg.ndim != 2:
            raise ValueError(f"被试{sub_id} {key} EEG维度异常: {eeg.shape}")

        # 保留前62个通道
        eeg = eeg[:N_CHANNELS_KEEP, :]
        n_channels, n_points = eeg.shape

        # EEG可整除的秒数
        n_seconds_eeg = n_points // SRATE
        if n_seconds_eeg == 0:
            print(f"  警告: 被试{sub_id} {key} 数据不足1秒，跳过")
            continue

        # ---------- 标签部分 ----------
        label_raw = label_mat[key]
        labels_2d = unify_label_shape(label_raw)  # (秒数, 2)
        n_seconds_label = labels_2d.shape[0]

        # EEG 与 标签长度不一致时，取最小值
        n_seconds = min(n_seconds_eeg, n_seconds_label)
        if n_seconds_eeg != n_seconds_label:
            print(f"  提示: 被试{sub_id} {key} EEG秒数={n_seconds_eeg}, "
                  f"标签秒数={n_seconds_label}, 使用 n_seconds={n_seconds}")

        # 截断EEG到对齐长度
        eeg = eeg[:, :n_seconds * SRATE]
        labels_2d = labels_2d[:n_seconds, :]

        # ---------- 5个频段滤波 & DE 特征 ----------
        band_de_list = []  # 每个元素形状 (n_seconds, 62)
        for (low, high) in BANDS:
            eeg_band = bandpass_filter(eeg, low, high, fs=SRATE, order=4)  # (62, T)
            de_feats = compute_de_features(eeg_band, n_seconds, fs=SRATE)  # (n_seconds, 62)
            band_de_list.append(de_feats)

        # band_de_list 有 5 个数组，每个 (n_seconds, 62)
        # 接下来按秒、按通道组织成 (62, 5)
        for sec in range(n_seconds):
            # 这一秒的 62 通道、5 频段DE
            feature_sec = np.zeros((N_CHANNELS_KEEP, len(BANDS)), dtype=np.float32)
            for b_idx, de_feats in enumerate(band_de_list):
                # de_feats[sec, :] -> (62,)
                feature_sec[:, b_idx] = de_feats[sec, :]

            # 对应标签
            val = labels_2d[sec, 0]
            aro = labels_2d[sec, 1]
            label = val_aro_to_label(val, aro)

            feature_list.append(feature_sec)         # (62, 5)
            labels_list.append(int(label))

    # ---------- 整合并保存 .mat ----------
    n_samples = len(feature_list)
    if n_samples == 0:
        print(f"被试{sub_id} 未生成任何样本，跳过保存")
        return

    # (n_samples, 62, 5)
    samples_arr = np.stack(feature_list, axis=0).astype(np.float32)
    # 展平成 (n_samples, 310 = 62*5)，按通道优先：
    # 每一行为 [ch1_5频段, ch2_5频段, ..., ch62_5频段]
    samples_flat = samples_arr.reshape(n_samples, -1)  # (n_samples, 310)

    labels_arr = np.array(labels_list, dtype=np.int32).reshape(-1, 1)

    # 构造 MATLAB 结构体 dataset_session1，含字段 feature 和 label
    dataset_session1 = {
        "feature": samples_flat,  # (n_samples, 310)
        "label": labels_arr       # (n_samples, 1)
    }

    # 输出文件名: sub_{id}_session_1.mat
    out_name = f"sub_{sub_id}_session_1.mat"
    out_path = os.path.join(OUT_DIR, out_name)
    sio.savemat(out_path, {
        "dataset_session1": dataset_session1
    }, do_compression=True)

    print(f"被试{sub_id} 处理完成 -> {out_path}, "
          f"样本数: {n_samples}, feature形状: {samples_flat.shape}")


def main():
    ensure_out_dir()
    for sub_id in range(1, N_SUBJECTS + 1):
        process_subject(sub_id)


if __name__ == "__main__":
    main()
