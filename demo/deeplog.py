#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
sys.path.append('../')

from logdeep.models.lstm import deeplog, loganomaly, robustlog
from logdeep.tools.predict import Predicter
from logdeep.tools.train import Trainer
from logdeep.tools.utils import *


# Config Parameters
# 这段代码是 Python 代码，它创建了一个空的字典。
# 在 Python 中，字典是一种可变的数据类型，用于存储键值对。
# 字典中的键必须是唯一的，而值可以是任何类型的对象。
# 在这段代码中，我们使用 dict() 函数创建了一个空的字典，然后将其赋值给名为 options 的变量。
# 这样，我们就可以向 options 字典中添加键值对，以便在程序的其他部分中使用它们
options = dict()
options['data_dir'] = '../data/'
options['window_size'] = 10
options['device'] = "cpu"

# Smaple
options['sample'] = "sliding_window"
options['window_size'] = 10  # if fix_window

# Features
# 这些参数的设置可以根据具体的深度学习模型和任务需求进行调整。
# 例如，如果输入数据包含序列信息、数值信息和语义信息，那么将相应的特征类型设置为True，以确保模型可以有效地利用这些信息。
# 这些参数的灵活性使得模型可以适应不同类型的输入数据。
# 序列特征（Sequentials）：表示模型是否使用序列型的特征。
# 如果设置为True，说明模型将考虑输入中的序列型特征。
options['sequentials'] = True
# 数值特征（Quantitatives）：表示模型是否使用数值型的特征。
# 如果设置为True，说明模型将考虑输入中的数值型特征。
options['quantitatives'] = False
# 语义特征（Semantics）：表示模型是否使用语义型的特征。
# 如果设置为True，说明模型将考虑输入中的语义型特征。
options['semantics'] = False
# 征数量（Feature Num）：通过将上述三种特征的设置相加，计算出模型总共考虑的特征数量。
# 这个值将用于定义模型的输入层
options['feature_num'] = sum(
    [options['sequentials'], options['quantitatives'], options['semantics']])

# Model
# 这些参数的设置通常依赖于具体的任务和数据。
# 通过调整这些参数，可以影响模型的容量、训练速度和泛化能力。
# 在实际应用中，这些参数可能需要通过实验和调整来找到最佳的组合，以获得最好的性能。
options['input_size'] = 1
# 含义：表示模型接受的每个输入序列的特征维度。
# 作用：指定输入数据的维度，对于时间序列数据，这可能是序列中每个时间步的特征数量
options['hidden_size'] = 64
# 含义：表示模型中隐藏层的神经元数量。
# 作用：决定了模型中每个时间步的隐藏状态的维度。隐藏状态是模型在处理序列时的内部表示
options['num_layers'] = 2
# 含义：表示模型中堆叠的RNN层的数量。
# 作用：可以通过堆叠多层RNN来增加模型的表示能力，更好地捕捉序列中的复杂模式。
options['num_classes'] = 28
# 含义：表示模型的输出类别数量。
# 作用：适用于分类任务，指定模型需要输出的类别数量。例如，如果是一个多类别分类问题，这个值表示模型需要输出的不同类别数量。

# Train

# batch_size 是深度学习中的一个超参数，它指定了在训练神经网络时一次性处理的样本数量。
# 在机器学习中，通常将训练数据集分成多个批次进行训练，每个批次包含 batch_size 个样本。
# 这样做的好处是可以利用矩阵运算的并行性，提高训练效率。
# 同时，较大的 batch_size 可以减少训练过程中的随机性，提高模型的稳定性和泛化能力。
# 但是，较大的 batch_size 也会占用更多的内存，可能会导致 GPU 内存不足，从而无法训练模型。
# 因此，batch_size 的大小需要根据具体的模型和硬件环境进行调整。

options['batch_size'] = 2048
options['accumulation_step'] = 1

# max_epoch 是一个超参数，它指定了训练神经网络的最大 epoch 数量。
# 在机器学习中，一个 epoch 指的是训练数据集中所有样本都被用于训练一次的过程。
# 在训练过程中，神经网络会根据训练数据集中的样本不断调整自己的权重和偏置，以便更好地拟合数据。
# max_epoch 的值通常是手动设置的，可以根据训练数据集的大小和模型的复杂度来调整。
# 如果 max_epoch 的值设置得太小，那么模型可能无法充分学习数据中的模式，导致欠拟合。
# 如果 max_epoch 的值设置得太大，那么模型可能会过度拟合训练数据，导致泛化性能下降。

# 优化器（Optimizer）：用于最小化训练损失的算法。
# 在这里，选择了Adam优化器，它是一种常用的自适应学习率优化算法，通常在深度学习中表现良好
options['optimizer'] = 'adam'
# 学习率（Learning Rate）：控制模型参数更新的步长。
# 这里设置学习率为0.001，表示在每次参数更新时，每个参数的值都会以0.001的步长进行调整。
options['lr'] = 0.001
# 最大训练轮数（Max Epoch）：训练过程中模型将遍历整个训练数据的次数。
# 这里设置最大训练轮数为370，表示模型将在训练数据上进行370次完整的训练。
options['max_epoch'] = 370
# 学习率衰减步骤（Learning Rate Decay Steps）：指定在哪些训练轮数时降低学习率。
# 在这里，学习率将在第300轮和第350轮时进行衰减
options['lr_step'] = (300, 350)
# 学习率衰减比例（Learning Rate Decay Ratio）：指定学习率在每个衰减步骤中的衰减比例。
# 在这里，学习率将以0.1的比例进行衰减，即乘以0.1。
options['lr_decay_ratio'] = 0.1 # 学习衰减率

options['resume_path'] = None
options['model_name'] = "deeplog"
options['save_dir'] = "../result/deeplog/"

# Predict
options['model_path'] = "../result/deeplog/deeplog_last.pth"
options['num_candidates'] = 9

seed_everything(seed=1234)


def train():
    Model = deeplog(input_size=options['input_size'],
                    hidden_size=options['hidden_size'],
                    num_layers=options['num_layers'],
                    num_keys=options['num_classes'])
    trainer = Trainer(Model, options)
    trainer.start_train()


def predict():
    Model = deeplog(input_size=options['input_size'],
                    hidden_size=options['hidden_size'],
                    num_layers=options['num_layers'],
                    num_keys=options['num_classes'])
    predicter = Predicter(Model, options)
    predicter.predict_unsupervised()


if __name__ == "__main__":
    # 这段代码是使用 Python 的 argparse 模块来解析命令行参数的。
    # argparse 模块可以让人轻松编写用户友好的命令行接口。
    # 程序定义它需要哪些参数，argparse 将会知道如何从 sys.argv 解析它们。
    # 在这段代码中，我们创建了一个 ArgumentParser 对象，然后使用 add_argument() 方法将单个参数规格说明关联到解析器。
    # 最后，我们调用 parse_args() 方法来解析命令行参数并将其转换为对象。
    # 这段代码中，我们定义了一个名为 mode 的位置参数，它只能取两个值：train 或 predict。
    # 如果用户输入的参数不在这两个值中，程序将会抛出错误。

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'predict'])
    # args = parser.parse_args()
    #if args.mode == 'train':
    train()
    #else:
     #   predict()
