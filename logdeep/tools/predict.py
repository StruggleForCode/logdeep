#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gc
import os
import sys
import time
from collections import Counter
sys.path.append('../../')

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from logdeep.dataset.log import log_dataset
from logdeep.dataset.sample import session_window
from logdeep.tools.utils import (save_parameters, seed_everything,
                                 train_val_split)


def generate(name):
    window_size = 10
    hdfs = {}
    length = 0
    with open('../data/hdfs/' + name, 'r') as f:
        for ln in f.readlines():
            ln = list(map(lambda n: n - 1, map(int, ln.strip().split())))
            ln = ln + [-1] * (window_size + 1 - len(ln))
            hdfs[tuple(ln)] = hdfs.get(tuple(ln), 0) + 1
            length += 1
    print('Number of sessions({}): {}'.format(name, len(hdfs)))
    return hdfs, length


class Predicter():
    def __init__(self, model, options):
        self.data_dir = options['data_dir']
        self.device = options['device']
        self.model = model
        self.model_path = options['model_path']
        self.window_size = options['window_size']
        self.num_candidates = options['num_candidates']
        self.num_classes = options['num_classes']
        self.input_size = options['input_size']
        self.sequentials = options['sequentials']
        self.quantitatives = options['quantitatives']
        self.semantics = options['semantics']
        self.batch_size = options['batch_size']

    # 这段代码定义了一个 predict_unsupervised 方法，用于在无监督场景下使用训练好的模型进行预测。
    # 主要的步骤包括加载模型权重，对测试数据进行预测，并计算模型性能指标，如精确度（Precision）、召回率（Recall）和 F1 分数。
    def predict_unsupervised(self):
        # 将模型移动到指定的计算设备上，加载训练好的模型权重，然后将模型设置为评估模式。
        model = self.model.to(self.device)
        model.load_state_dict(torch.load(self.model_path)['state_dict'])
        model.eval()
        print('model_path: {}'.format(self.model_path))
        # 生成测试正常和异常数据的 DataLoader，并获取相应的数据长度。
        test_normal_loader, test_normal_length = generate('hdfs_test_normal')
        test_abnormal_loader, test_abnormal_length = generate(
            'hdfs_test_abnormal')
        # 初始化真正例（TP）和假正例（FP）的计数。
        TP = 0
        FP = 0
        # Test the model
        start_time = time.time()
        # 对测试集中的正常数据进行预测。
        # 对于每个数据序列，将窗口大小内的数据作为输入，进行模型预测，并比较真实标签和模型预测的结果。
        # 如果真实标签不在预测结果中，则将该样本计为假正例（FP）。
        with torch.no_grad():
            for line in tqdm(test_normal_loader.keys()):
                for i in range(len(line) - self.window_size):
                    seq0 = line[i:i + self.window_size]
                    label = line[i + self.window_size]
                    seq1 = [0] * 28
                    log_conuter = Counter(seq0)
                    for key in log_conuter:
                        seq1[key] = log_conuter[key]

                    seq0 = torch.tensor(seq0, dtype=torch.float).view(
                        -1, self.window_size, self.input_size).to(self.device)
                    seq1 = torch.tensor(seq1, dtype=torch.float).view(
                        -1, self.num_classes, self.input_size).to(self.device)
                    label = torch.tensor(label).view(-1).to(self.device)
                    output = model(features=[seq0, seq1], device=self.device)
                    predicted = torch.argsort(output,
                                              1)[0][-self.num_candidates:]
                    if label not in predicted:
                        FP += test_normal_loader[line]
                        break
        # 对测试集中的异常数据进行预测。
        # 对于每个数据序列，将窗口大小内的数据作为输入，进行模型预测，并比较真实标签和模型预测的结果。
        # 如果真实标签不在预测结果中，则将该样本计为真正例（TP）。
        with torch.no_grad():
            for line in tqdm(test_abnormal_loader.keys()):
                for i in range(len(line) - self.window_size):
                    seq0 = line[i:i + self.window_size]
                    label = line[i + self.window_size]
                    seq1 = [0] * 28
                    log_conuter = Counter(seq0)
                    for key in log_conuter:
                        seq1[key] = log_conuter[key]

                    seq0 = torch.tensor(seq0, dtype=torch.float).view(
                        -1, self.window_size, self.input_size).to(self.device)
                    seq1 = torch.tensor(seq1, dtype=torch.float).view(
                        -1, self.num_classes, self.input_size).to(self.device)
                    label = torch.tensor(label).view(-1).to(self.device)
                    output = model(features=[seq0, seq1], device=self.device)
                    predicted = torch.argsort(output,
                                              1)[0][-self.num_candidates:]
                    if label not in predicted:
                        TP += test_abnormal_loader[line]
                        break

        # 根据计算的真正例（TP）、假正例（FP）和假负例（FN），计算精确度（Precision）、召回率（Recall）和 F1 分数。
        # Compute precision, recall and F1-measure
        FN = test_abnormal_length - TP
        P = 100 * TP / (TP + FP)
        R = 100 * TP / (TP + FN)
        F1 = 2 * P * R / (P + R)
        # 输出计算得到的性能指标。
        print(
            'false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'
            .format(FP, FN, P, R, F1))
        print('Finished Predicting')
        # 输出完成预测的信息，并计算预测所花费的时间。
        elapsed_time = time.time() - start_time
        print('elapsed_time: {}'.format(elapsed_time))

    # 这段代码定义了 predict_supervised 方法，用于在监督学习场景下使用训练好的模型进行预测。
    # 与无监督学习场景不同，这里的预测是基于有标签的测试数据进行的，
    # 计算了一些监督学习的性能指标，包括精确度（Precision）、召回率（Recall）和 F1 分数。
    # 这个方法用于在监督学习场景中对测试数据进行预测，并评估模型的性能。
    def predict_supervised(self):
        # 将模型移动到指定的计算设备上，加载训练好的模型权重，然后将模型设置为评估模式。
        model = self.model.to(self.device)
        model.load_state_dict(torch.load(self.model_path)['state_dict'])
        model.eval()
        print('model_path: {}'.format(self.model_path))
        # 使用 session_window 函数生成测试数据，包括序列数据、数量数据和语义数据。
        test_logs, test_labels = session_window(self.data_dir, datatype='test')
        # 使用 log_dataset 类创建测试数据集，并构建 DataLoader 用于批量加载数据。
        test_dataset = log_dataset(logs=test_logs,
                                   labels=test_labels,
                                   seq=self.sequentials,
                                   quan=self.quantitatives,
                                   sem=self.semantics)
        self.test_loader = DataLoader(test_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=False,
                                      pin_memory=True)
        tbar = tqdm(self.test_loader, desc="\r")
        # 初始化真正例（TP）、假正例（FP）、假负例（FN）和真负例（TN）的计数。
        TP, FP, FN, TN = 0, 0, 0, 0
        # 遍历测试数据集中的每个批次。
        for i, (log, label) in enumerate(tbar):
            # 将测试数据的特征移动到指定的计算设备上，使用模型进行前向传播，并对输出进行阈值处理，生成预测结果。
            features = []
            for value in log.values():
                features.append(value.clone().to(self.device))
            output = self.model(features=features, device=self.device)
            output = F.sigmoid(output)[:, 0].cpu().detach().numpy()
            # predicted = torch.argmax(output, dim=1).cpu().numpy()
            predicted = (output < 0.2).astype(int)
            label = np.array([y.cpu() for y in label])
            # 根据计算的真正例（TP）、假正例（FP）、假负例（FN）和真负例（TN），计算精确度（Precision）、召回率（Recall）和 F1 分数。
            TP += ((predicted == 1) * (label == 1)).sum()
            FP += ((predicted == 1) * (label == 0)).sum()
            FN += ((predicted == 0) * (label == 1)).sum()
            TN += ((predicted == 0) * (label == 0)).sum()
        P = 100 * TP / (TP + FP)
        R = 100 * TP / (TP + FN)
        F1 = 2 * P * R / (P + R)
        print(
            'false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'
            .format(FP, FN, P, R, F1))
