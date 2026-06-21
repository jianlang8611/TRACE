# TRACEing the Shortcut: Mitigating Event-Level Spurious Correlations in Video Misinformation Detection

This repo provides a official implementation for paper: *TRACEing the Shortcut: Mitigating Event-Level Spurious
Correlations in Video Misinformation Detection*.

## Source Code Structure

```sh
├── data    # dataset path
│   ├── FakeSV
│   ├── FakeTT
│   └── FVC
├── src         # code of model arch and training
│   ├── main.py     # main code for training 
│   ├── model
│   │   ├──Base
│   │   ├──SVFEND    # implementation of SVFEND w/ TRACE
│   │   └──trace.py  # main code for our propose TRACE
└── └── utils
```

## Dataset

We provide video IDs for each dataset splits. Due to copyright restrictions, the raw datasets are not included. You can obtain the datasets from their respective original project sites.

+ [FakeSV](https://github.com/ICTMCG/FakeSV)
+ [FakeTT](https://github.com/ICTMCG/FakingRecipe)
+ [FVC](https://github.com/MKLab-ITI/fake-video-corpus)

## Usage

### Requirement

To set up the environment, run the following commands:

```sh
conda create --name TRACE python=3.10
conda activate TRACE
pip install -r requirements.txt
```

### Preprocess

Since TRACE is a plug-in framework, we do not adopt a unified preprocessing pipeline. Instead, we do follow the preprocessing procedures of each base detector to ensure a fair comparison.

### Run
```sh
python src/main.py --config-name SVFEND_FakeSV.yaml     # run SVFEND w/ TRACE on FakeSV
python src/main.py --config-name SVFEND_FakeTT.yaml     # run SVFEND w/ TRACE on FakeTT
python src/main.py --config-name SVFEND_FVC.yaml        # run SVFEND w/ TRACE on FVC
```
