# CCAP-Net
This is the source code of our paper 'Cross subject EEG emotion recognition via global and local domain alignment'.

## Project structure
```
CCAP-Net/
├─ data/                # Data directory
│  ├─ REFED_feature/    # REFED dataset feature data folder
│  ├─ SEED/             # SEED dataset feature data folder
│  ├─ SEED-IV/          # SEED-IV dataset feature data folder
│  ├─ TYUT3.0_feature/  # ENTER dataset feature data folder
│  ├─ ENTER_dataset_data_processing_integration.py  # ENTER dataset processing script
│  ├─ REFED_dataset_data_processing_integration.py  # REFED dataset processing script
│  ├─ SEED_dataset_data_processing_integration.py   # SEED dataset processing script
│  └─ SEED-IV_dataset_data_processing_integration.py# SEED-IV dataset processing script
├─ results/             # Model training results output directory
│  ├─ logs/             # Training logs folder
│  ├─ models/           # Model saving folder
│  ├─ visualization_results/ # Visualization results folder
│  ├─ SEED/             # SEED experiment results folder
│  ├─ SEED-IV/          # SEED-IV experiment results folder
│  ├─ ENTER/            # ENTER experiment results folder
│  └─ REFED/            # REFED experiment results folder
├─ ccap_net_fixed.py    # CCAP-Net model script
├─ cdan_fixed.py        # CDAN module script
├─ cscfa.py             # CSCFA module script
├─ train_fixed.py       # Model training script
└─ utils.py             # Utility functions script
```

## Preliminaries
Prepare dataset: SEED,SEED-IV,ENTER and REFED.The input features of the model can be obtained by running the 4 data processing scripts provided in the data folder. The project has also directly provided the model input features corresponding to the four data sets.

## Requirements
- Python 3.9+
- Install the corresponding software package in requirements.txt

## Usage
run the main function in the train.py

## Training model
- python train.py --dataset SEED --sessions 1 2 3
- python train.py --dataset SEED-IV --sessions 1 2 3
- python train.py --dataset ENTER --sessions 1
- python train.py --dataset REFED --sessions 1
