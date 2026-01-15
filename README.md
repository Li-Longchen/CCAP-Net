# CCAP-Net
This is the source code of our paper 'Cross subject EEG emotion recognition via global and local domain alignment'.

## project structure
```
CCAP-Net/
├── data/                    # 数据目录
│   ├── SEED/               # SEED数据集 (3类)
│   └── SEED-IV/            # SEED-IV数据集 (4类)
├── src/                     # 源代码
│   ├── models/             # 模型定义
│   ├── modules/            # 功能模块
│   ├── utils/              # 工具函数
│   ├── train.py            # 训练脚本
│   └── evaluate.py         # 评估脚本
├── results/                 # 输出结果
│   ├── models/             # 保存的模型
│   ├── logs/               # 训练日志
│   └── visualization_results/            # 生成的可视化图表
└── scripts/                 # 运行脚本
```

## requirements
- Python 3.9+
- PyTorch 2.0+
- NumPy, SciPy, scikit-learn

## usage
After modify setting (path, etc), just run the main function in the train.py

### training model
python train.py --dataset SEED --sessions 1 2 3
python train.py --dataset SEED-IV --sessions 1 2 3

### evaluating model
python evaluate.py --dataset SEED --sessions 1 2 3
