# ES-TFI
Code for the paper "ES-TFI: Evolutionary Search for Task-Specific Feature Interactions in Multi-Task Recommendation".

## Requirements

Ensure that Python and PyTorch are installed in your environment.  
Our experiments were conducted with the following configuration:

- Python 3.8
- PyTorch 2.4.1

If you plan to leverage GPU acceleration, please install the appropriate **CUDA** version compatible with your PyTorch installation.

Our implementation is built upon the [DeepCTR-Torch](https://github.com/shenweichen/DeepCTR-Torch) framework(v0.2.9). Install through `pip install -U deepctr-torch`

## Datasets

Please download the datasets from the following official sources:

- KuaiRand-Pure : https://kuairand.com/
- QB-Video: https://static.qblv.qq.com/qblv/h5/algo-frontend/tenrec_dataset.html
- AliCCP: https://tianchi.aliyun.com/dataset/408

After downloading the datasets, place them in the corresponding data directory following the project structure.

## Example Usage

After obtaining the source code and preparing the datasets, you can train the **ES-TFI** model using the following commands:

```
cd main
python train.py --dataset qb_video --gpu 0
```

The training process will automatically record the evolutionary search process in the `param/` and final results in the `result/` directory.

## Code Availability

This repository currently provides a simplified version of the ES-TFI framework for reference.

The complete implementation will be released after the paper is accepted.
