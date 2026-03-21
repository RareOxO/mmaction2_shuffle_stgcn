# Shuffle-STGCN

基于 [MMAction2](https://github.com/open-mmlab/mmaction2) 框架，在 **ST-GCN / ST-GCN++** 算法基础上的改进工作

## Overview

本项目将 **ShuffleNet**（Zhang et al., CVPR 2018, [arXiv:1801.07455](https://arxiv.org/abs/1801.07455)）的通道混洗（Channel Shuffle）机制引入动作识别的时序卷积模块（TCN），提出 **Shuffle-TCN**，替换原始 ST-GCN++ 中的 MS-TCN 模块。

## Shuffle-TCN 模块详解

**Shuffle-TCN 模块位置：** `mmaction/models/utils/gcn_utils.py`，class `shuffle_tcn`

### 网络结构图

Channel Shuffle Module

<img src="resources/channel_shuffle.png" alt="Channel Shuffle" width="40%">

Downsample Module

<img src="resources/downsampling.png" alt="Downsample" width="50%">

## 配置文件

| 文件 | 说明 |
|------|------|
| `configs/skeleton/shuffle_stgcn/shuffle_stgcn.py` | 基础配置（模型结构定义） |
| `configs/skeleton/shuffle_stgcn/shuffle_stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py` | 完整训练配置 |

## 开启训练

```bash
bash shuffle_stgcn_train.sh
```

## References

- **ST-GCN**: Yan et al., *Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition*, AAAI 2018
- **ST-GCN++**: Duan et al., *Revisiting Skeleton-Based Action Recognition*, CVPR 2022
- **ShuffleNet**: Zhang et al., *ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Applications*, CVPR 2018 ([arXiv:1801.07455](https://arxiv.org/abs/1801.07455))
- **MMAction2**: [https://github.com/open-mmlab/mmaction2](https://github.com/open-mmlab/mmaction2)
