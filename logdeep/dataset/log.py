#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

# 这段代码定义了一个名为 `log_dataset` 的类，它是 PyTorch 中的 `Dataset` 类的子类，用于创建一个自定义的数据集。
# 这样的数据集可以传递给 PyTorch 的数据加载器（`DataLoader`），用于在训练模型时提供批量的数据。
# 这个自定义的数据集类的目的是将原始的日志数据和标签组织成 PyTorch 模型可以处理的形式。
# 通过指定在实例化时是否包含序列、数量、和语义特征，可以根据模型的需求动态地选择性地使用这些特征。
# 在训练时，可以将该数据集传递给 PyTorch 的数据加载器，实现批量数据的加载。
class log_dataset(Dataset):
    # 构造函数接受四个参数：
    #    - `logs`: 包含日志数据的字典，包括三个键：'Sequentials'、'Quantitatives'、'Semantics'。
    #    - `labels`: 包含每个日志序列标签的列表。
    #    - `seq`, `quan`, `sem`: 用于控制是否包含序列、数量、和语义特征，默认值为 True。
    #    在构造函数内部，它根据参数的值决定是否保存对应的特征，以及它们的名称。
    def __init__(self, logs, labels, seq=True, quan=False, sem=False):
        self.seq = seq
        self.quan = quan
        self.sem = sem
        if self.seq:
            self.Sequentials = logs['Sequentials']
        if self.quan:
            self.Quantitatives = logs['Quantitatives']
        if self.sem:
            self.Semantics = logs['Semantics']
        self.labels = labels

    #  实现了 `Dataset` 类的 `__len__` 方法，返回数据集的总长度，即标签的数量。
    def __len__(self):
        return len(self.labels)

    #  实现了 `Dataset` 类的 `__getitem__` 方法，该方法根据给定的索引 `idx` 返回一个数据样本。
    #    - 创建一个字典 `log` 用于存储特征和标签。
    #    - 根据 `seq`, `quan`, `sem` 的值，将对应的特征转换为 PyTorch 的张量，并存储在 `log` 字典中。
    #    - 返回 `log` 字典和对应索引的标签。
    def __getitem__(self, idx):
        log = dict()
        if self.seq:
            log['Sequentials'] = torch.tensor(self.Sequentials[idx],
                                              dtype=torch.float)
        if self.quan:
            log['Quantitatives'] = torch.tensor(self.Quantitatives[idx],
                                                dtype=torch.float)
        if self.sem:
            log['Semantics'] = torch.tensor(self.Semantics[idx],
                                            dtype=torch.float)
        return log, self.labels[idx]


if __name__ == '__main__':
    data_dir = '../../data/hdfs/hdfs_train'
    window_size = 10
    train_logs = prepare_log(data_dir=data_dir,
                             datatype='train',
                             window_size=window_size)
    train_dataset = log_dataset(log=train_logs, seq=True, quan=True)
    print(train_dataset[0])
    print(train_dataset[100])
