import os
import scipy.io as sio
import numpy as np

# 定义数据集的路径
base_path = r'E:\脑电信号数据集\SEED数据集\ExtractedFeatures'
feature_path = os.path.join(base_path, 'SEED_feature')
os.makedirs(os.path.join(feature_path, 'session1'), exist_ok=True)
os.makedirs(os.path.join(feature_path, 'session2'), exist_ok=True)
os.makedirs(os.path.join(feature_path, 'session3'), exist_ok=True)

# 定义每个 session 的被试文件名
session1_files = [
    '1_20131027.mat', '2_20140404.mat', '3_20140603.mat', '4_20140621.mat', '5_20140411.mat',
    '6_20130712.mat', '7_20131027.mat', '8_20140511.mat', '9_20140620.mat', '10_20131130.mat',
    '11_20140618.mat', '12_20131127.mat', '13_20140527.mat', '14_20140601.mat', '15_20130709.mat'
]

session2_files = [
    '1_20131030.mat', '2_20140413.mat', '3_20140611.mat', '4_20140702.mat', '5_20140418.mat',
    '6_20131016.mat', '7_20131030.mat', '8_20140514.mat', '9_20140627.mat', '10_20131204.mat',
    '11_20140625.mat', '12_20131201.mat', '13_20140603.mat', '14_20140615.mat', '15_20131016.mat'
]

session3_files = [
    '1_20131107.mat', '2_20140419.mat', '3_20140629.mat', '4_20140705.mat', '5_20140506.mat',
    '6_20131113.mat', '7_20131106.mat', '8_20140521.mat', '9_20140704.mat', '10_20131211.mat',
    '11_20140630.mat', '12_20131207.mat', '13_20140610.mat', '14_20140627.mat', '15_20131105.mat'
]

# 定义从每个mat文件中提取的DE特征变量
de_features = ['de_LDS1', 'de_LDS2', 'de_LDS3', 'de_LDS4', 'de_LDS5', 'de_LDS6', 'de_LDS7',
               'de_LDS8', 'de_LDS9', 'de_LDS10', 'de_LDS11', 'de_LDS12', 'de_LDS13', 'de_LDS14',
               'de_LDS15']

# 函数：提取并保存DE特征
def process_session(session_files, session_num):
    for file_name in session_files:
        # 加载每个被试的.mat文件
        mat_data = sio.loadmat(os.path.join(base_path, file_name))

        # 获取DE特征数据和标签
        de_data = [mat_data[de_feature] for de_feature in de_features]
        labels = sio.loadmat(os.path.join(base_path, 'label.mat'))['label']

        # 每个被试的特征数据处理
        all_features = []
        all_labels = []

        for i in range(15):
            de = de_data[i]  # 每个试验的DE特征，形状：62x样本数x5
            label = labels[0][i]  # 获取当前试验的标签

            # 将DE特征重塑为样本数x310的形状
            reshaped_de = de.reshape((-1, 310))  # 将62x样本数x5 reshape为 样本数x310

            # 每个试验的数据加入到总数据中
            all_features.append(reshaped_de)
            all_labels.append(np.full((reshaped_de.shape[0], 1), label))

        # 将所有试验的特征和标签合并
        all_features = np.vstack(all_features)
        all_labels = np.vstack(all_labels)

        # 保存到.mat文件中
        subject_num = int(file_name.split('_')[0])  # 获取被试编号
        save_path = os.path.join(feature_path, f'session{session_num}', f'sub_{subject_num}_session_{session_num}.mat')

        sio.savemat(save_path, {f'dataset_session{session_num}': {'feature': all_features, 'label': all_labels}})
        print(f"Saved session {session_num} subject {subject_num} data to {save_path}")

# 处理每个session的数据
process_session(session1_files, 1)
process_session(session2_files, 2)
process_session(session3_files, 3)

print("Data processing complete.")
