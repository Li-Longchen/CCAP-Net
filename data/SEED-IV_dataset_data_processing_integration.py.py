import os
import scipy.io as sio
import numpy as np

# 定义数据集的路径
base_path = r'E:\脑电信号数据集\SEED-IV数据集\eeg_feature_smooth'
feature_path = os.path.join(base_path, 'SEEDIV_feature')
os.makedirs(os.path.join(feature_path, 'session1'), exist_ok=True)
os.makedirs(os.path.join(feature_path, 'session2'), exist_ok=True)
os.makedirs(os.path.join(feature_path, 'session3'), exist_ok=True)

# session1的文件名
session1_files = [
    '1_20160518.mat', '2_20150915.mat', '3_20150919.mat', '4_20151111.mat', '5_20160406.mat',
    '6_20150507.mat', '7_20150715.mat', '8_20151103.mat', '9_20151028.mat', '10_20151014.mat',
    '11_20150916.mat', '12_20150725.mat', '13_20151115.mat', '14_20151205.mat', '15_20150508.mat'
]

# session2的文件名
session2_files = [
    '1_20161125.mat', '2_20150920.mat', '3_20151018.mat', '4_20151118.mat', '5_20160413.mat',
    '6_20150511.mat', '7_20150717.mat', '8_20151110.mat', '9_20151119.mat', '10_20151021.mat',
    '11_20150921.mat', '12_20150804.mat', '13_20151125.mat', '14_20151208.mat', '15_20150514.mat'
]

# session3的文件名
session3_files = [
    '1_20161126.mat', '2_20151012.mat', '3_20151101.mat', '4_20151123.mat', '5_20160420.mat',
    '6_20150512.mat', '7_20150721.mat', '8_20151117.mat', '9_20151209.mat', '10_20151023.mat',
    '11_20151011.mat', '12_20150807.mat', '13_20161130.mat', '14_20151215.mat', '15_20150527.mat'
]

# 定义每个 session 的标签
session1_label = [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3]
session2_label = [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1]
session3_label = [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]

# DE特征变量
de_features = [f'de_LDS{i + 1}' for i in range(24)]  # 24个试验对应的变量


# 提取并保存DE特征的函数
def process_session(session_files, session_num, labels):
    for file_name in session_files:
        # 加载每个被试的.mat文件
        mat_data = sio.loadmat(os.path.join(base_path, str(session_num), file_name))

        # 获取DE特征数据
        de_data = [mat_data[de_feature] for de_feature in de_features]

        # 获取当前session对应的标签
        session_labels = np.array(labels)

        # 每个被试的特征数据处理
        all_features = []
        all_labels = []

        for i in range(24):
            de = de_data[i]  # 每个试验的DE特征，形状：62x样本数x5
            label = session_labels[i]  # 获取当前试验的标签

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
process_session(session1_files, 1, session1_label)
process_session(session2_files, 2, session2_label)
process_session(session3_files, 3, session3_label)

print("Data processing complete.")
