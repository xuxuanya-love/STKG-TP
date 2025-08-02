## Overview
<img width="1157" alt="image" src="https://github.com/user-attachments/assets/a3504dd6-e7e4-4133-aa07-f9cf214a05e1" />

## Dependencies

  - python=3.10
  - cudatoolkit=12.1
  - torch=2.0.1
  - torchvision=0.17.1
  - torchaudio=2.2.1
  - torch-cluster=1.6.3
  - torch-geometric=2.5.3
  - torch-scatter=2.1.2
  - torch-sparse=0.6.18
  - torch-spline-conv=1.2.2
  - scikit-learn=1.5.0
  - numpy=1.22.4

## Explain
split_data -- Time window segmentation from EEG signals

loadData -- Load input data for the model (EEG features and brain region node embeddings)

CNN -- Perform spatiotemporal convolution to extract features

GCN -- Perform spatiotemporal convolution

GRU -- Perform sequence operations on spatiotemporal convolution features

train -- Connect the entire model in series

BrainRegions -- Compute and visualize the brain regions

Add_Windows -- Apply windowing

## Run
Loss Visualize -- tensorboard --logdir='.\path\to\log' --host=127.0.0.1 --port=8008

Train -- train STKG.py # STKG Module

Train -- train TP.py # TP Module

Fusion -- train Cross_Fusion.PY # Cross_Fusion model

