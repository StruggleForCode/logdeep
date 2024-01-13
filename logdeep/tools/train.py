#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gc
import os
import sys
import time
sys.path.append('../../')

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from logdeep.dataset.log import log_dataset
from logdeep.dataset.sample import sliding_window, session_window
from logdeep.tools.utils import (save_parameters, seed_everything,
                                 train_val_split)


class Trainer():
    def __init__(self, model, options):
        self.model_name = options['model_name']
        self.save_dir = options['save_dir']
        self.data_dir = options['data_dir']
        self.window_size = options['window_size']
        self.batch_size = options['batch_size']

        self.device = options['device']
        self.lr_step = options['lr_step']
        self.lr_decay_ratio = options['lr_decay_ratio']
        self.accumulation_step = options['accumulation_step']
        self.max_epoch = options['max_epoch']

        self.sequentials = options['sequentials']
        self.quantitatives = options['quantitatives']
        self.semantics = options['semantics']
        self.sample = options['sample']
        self.feature_num = options['feature_num']

        # 这段代码的作用是在文件系统中创建目录（文件夹），并且如果目录已经存在，不会引发错误
        # os.makedirs 是一个用于递归创建目录的函数，可以创建多层嵌套的目录结构。
        # self.save_dir 是目录的路径，这个路径通常是在训练过程中保存模型、日志等文件的根目录。
        # exist_ok=True 参数表示，如果目录已经存在，不会引发 FileExistsError 错误。如果设置为 False（默认值），如果目录已经存在，将会引发错误。
        os.makedirs(self.save_dir, exist_ok=True)
        if self.sample == 'sliding_window':
            train_logs, train_labels = sliding_window(self.data_dir,
                                                  datatype='train',
                                                  window_size=self.window_size)
            val_logs, val_labels = sliding_window(self.data_dir,
                                              datatype='val',
                                              window_size=self.window_size,
                                              sample_ratio=0.001)
        elif self.sample == 'session_window':
            train_logs, train_labels = session_window(self.data_dir,
                                                      datatype='train')
            val_logs, val_labels = session_window(self.data_dir,
                                                  datatype='val')
        else:
            raise NotImplementedError

        train_dataset = log_dataset(logs=train_logs,
                                    labels=train_labels,
                                    seq=self.sequentials,
                                    quan=self.quantitatives,
                                    sem=self.semantics)
        valid_dataset = log_dataset(logs=val_logs,
                                    labels=val_labels,
                                    seq=self.sequentials,
                                    quan=self.quantitatives,
                                    sem=self.semantics)
        # del 删除变量
        del train_logs
        del val_logs
        # gc.collect() 是 Python 中的一种语句，用于手动触发垃圾回收机制。
        # Python 的垃圾回收机制是自动的，但是在某些情况下，手动触发垃圾回收机制可以释放内存并提高程序的性能。
        # gc.collect() 函数可以接受一个参数 generation，用于指定回收的代数。
        # 如果不传入参数，则默认回收所有代数的垃圾对象。
        gc.collect()

        # 这部分代码创建了两个 PyTorch 的数据加载器 (DataLoader) 对象，
        # 分别用于训练数据集 (train_dataset) 和验证数据集 (valid_dataset)。
        # 数据加载器用于批量地加载数据，适用于训练深度学习模型。

        # 创建了一个名为 train_loader 的训练数据加载器。参数说明如下：
        # train_dataset: 训练数据集，这是之前定义的 log_dataset 的一个实例。
        # batch_size: 指定每个批次的样本数量。
        # shuffle=True: 表示在每个 epoch 开始时对数据进行洗牌，有助于模型学习更好的特征。
        # pin_memory=True: 如果为True，数据加载器会将数据加载到 CUDA 固定内存区域中，可以加速数据传输到 GPU。
        self.train_loader = DataLoader(train_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=True,
                                       pin_memory=True)
        # 创建了一个名为 valid_loader 的验证数据加载器，使用了相同的参数设置，除了 shuffle=False，因为在验证阶段不需要对数据进行洗牌。
        # 这两个数据加载器可以在模型的训练和验证阶段使用，通过迭代它们，可以获取每个批次的输入数据和对应的标签。
        # 这样，你可以方便地使用 PyTorch 提供的训练和验证循环来训练和评估你的深度学习模型。
        self.valid_loader = DataLoader(valid_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=False,
                                       pin_memory=True)

        self.num_train_log = len(train_dataset)
        self.num_valid_log = len(valid_dataset)

        print('Find %d train logs, %d validation logs' %
              (self.num_train_log, self.num_valid_log))
        print('Train batch size %d ,Validation batch size %d' %
              (options['batch_size'], options['batch_size']))

        # 这一行代码将之前创建的深度学习模型 model 移动到指定的计算设备 self.device 上。
        # 在深度学习中，模型的训练和推理通常是在 GPU 上进行的，因为 GPU 具有并行计算的能力，可以显著加速训练过程。
        # 具体来说，这行代码使用 PyTorch 中的 to 方法将模型移动到指定的设备。
        # self.device 可以是 CPU 或 GPU。如果 self.device 是 GPU，模型将被移动到 GPU 上进行计算
        self.model = model.to(self.device)

        # This is test github connect
        # SGD 优化器，
        # Adam 优化器
        # self.model.parameters() 是模型的参数，lr 是学习率，momentum 和 betas 是优化器的超参数，用于控制优化器的行为。
        # SGD 优化器使用动量来加速梯度下降的过程，而 Adam 优化器则使用自适应学习率来调整每个参数的更新步长
        # 如果选项中的优化器是 'sgd'，则实例化一个 SGD 优化器。
        # 这里使用了模型的参数 self.model.parameters() 作为优化器的参数，学习率 lr 由选项中提供，而动量设置为0.9。
        if options['optimizer'] == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=options['lr'],
                                             momentum=0.9)
        elif options['optimizer'] == 'adam':
            # 如果选项中的优化器是 'adam'，则实例化一个 Adam 优化器。
            # 同样，使用了模型的参数 self.model.parameters() 作为优化器的参数，学习率 lr 由选项中提供，而 betas 参数设置为(0.9, 0.999)。
            # 通过这个代码段，你可以在训练过程中选择使用 SGD 或 Adam 优化器，并通过设置不同的学习率和参数来调整优化器的行为。
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=options['lr'],
                betas=(0.9, 0.999),
            )
        else:
            raise NotImplementedError

        self.start_epoch = 0
        self.best_loss = 1e10
        self.best_score = -1
        save_parameters(options, self.save_dir + "parameters.txt")
        self.log = {
            "train": {key: []
                      for key in ["epoch", "lr", "time", "loss"]},
            "valid": {key: []
                      for key in ["epoch", "lr", "time", "loss"]}
        }
        # 如果检查点文件存在，调用 self.resume 方法来加载检查点。
        # load_optimizer=True 表示同时加载优化器的状态，这样可以在之前训练的基础上继续进行训练
        if options['resume_path'] is not None:
            if os.path.isfile(options['resume_path']):
                self.resume(options['resume_path'], load_optimizer=True)
            else:
                print("Checkpoint not found")

    # 这段代码定义了一个 resume 方法，用于从之前保存的模型检查点文件中恢复训练的状态。
    # 主要的功能是加载模型的参数、优化器的状态、训练的轮数等信息
    def resume(self, path, load_optimizer=True):
        print("Resuming from {}".format(path))
        # 使用 PyTorch 的 torch.load 方法加载检查点文件，得到一个包含各种信息的字典 checkpoint
        checkpoint = torch.load(path)
        # self.start_epoch: 更新为检查点中记录的 epoch 数加1，表示从下一个 epoch 开始训练。
        # self.best_loss: 更新为检查点中记录的最佳损失值。
        # self.log: 更新为检查点中记录的日志信息。
        # self.best_f1_score: 更新为检查点中记录的最佳 F1 分数。
        # 使用 load_state_dict 方法加载模型的参数。
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_loss = checkpoint['best_loss']
        self.log = checkpoint['log']
        self.best_f1_score = checkpoint['best_f1_score']
        self.model.load_state_dict(checkpoint['state_dict'])
        # 如果检查点中包含优化器的状态信息，并且 load_optimizer 参数为 True，则加载优化器的状态。
        # 这个方法的目的是将模型和优化器的状态从之前保存的检查点文件中还原，以便从上次中断的地方继续训练。
        if "optimizer" in checkpoint.keys() and load_optimizer:
            print("Loading optimizer state dict")
            self.optimizer.load_state_dict(checkpoint['optimizer'])

    # 这段代码定义了 save_checkpoint 方法，用于保存模型的检查点，包括模型的参数、优化器的状态（可选）、当前最佳损失、训练日志等信息。
    # 这个方法用于在训练过程中保存模型的状态，以便在需要时可以从保存的地方继续训练，或者用于在验证集上选择性地保存表现最好的模型。
    def save_checkpoint(self, epoch, save_optimizer=True, suffix=""):
        # 构建一个字典 checkpoint 包含以下信息：
        # epoch: 当前 epoch 数。
        # state_dict: 模型的参数字典。
        # best_loss: 当前最佳的训练损失。
        # log: 训练过程中的日志信息。
        # best_score: 可能表示模型在验证集上的最佳性能指标。
        checkpoint = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "best_loss": self.best_loss,
            "log": self.log,
            "best_score": self.best_score
        }
        # 如果 save_optimizer 参数为 True，则将优化器的状态字典添加到检查点中。
        if save_optimizer:
            checkpoint['optimizer'] = self.optimizer.state_dict()
        # 构建保存模型检查点的路径，包括目录、模型名称和指定的后缀。
        save_path = self.save_dir + self.model_name + "_" + suffix + ".pth"
        # 使用 PyTorch 的 torch.save 方法将构建的检查点保存到指定的路径。
        torch.save(checkpoint, save_path)
        print("Save model checkpoint at {}".format(save_path))

    def save_log(self):
        try:
            for key, values in self.log.items():
                pd.DataFrame(values).to_csv(self.save_dir + key + "_log.csv",
                                            index=False)
            print("Log saved")
        except:
            print("Failed to save logs")

    def train(self, epoch):
        # 将当前 epoch 的信息添加到训练日志中。
        self.log['train']['epoch'].append(epoch)
        start = time.strftime("%H:%M:%S")
        # 从优化器的状态字典中获取当前学习率。
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        # 输出当前 epoch 的开始信息，包括 epoch 数、开始时间、学习率等。
        print("Starting epoch: %d | phase: train | ⏰: %s | Learning rate: %f" %
              (epoch, start, lr))
        # 将当前 epoch 的学习率记录到训练日志中。
        self.log['train']['lr'].append(lr)
        self.log['train']['time'].append(start)
        # 将模型设置为训练模式，这会启用 dropout 等训练时的特定操作。
        self.model.train()
        # 将优化器的梯度清零，准备进行新一轮的梯度计算和优化。
        self.optimizer.zero_grad()
        # 定义交叉熵损失函数，用于计算模型的输出和真实标签之间的损失。
        criterion = nn.CrossEntropyLoss()
        # 使用 tqdm 创建一个进度条，用于可视化训练进度。
        tbar = tqdm(self.train_loader, desc="\r")
        num_batch = len(self.train_loader)
        total_losses = 0
        for i, (log, label) in enumerate(tbar):
            # 遍历训练数据集中的每个批次。
            # 将当前批次的日志数据中的每个特征转换为 PyTorch 张量，并移动到指定的计算设备上。
            features = []
            for value in log.values():
                features.append(value.clone().detach().to(self.device))
            # 使用模型进行前向传播，得到模型的输出。
            output = self.model(features=features, device=self.device)
            # 计算模型输出与真实标签之间的损失。
            loss = criterion(output, label.to(self.device))
            # 累积当前 epoch 的总损失。
            total_losses += float(loss)
            # 将损失除以累积步数，执行反向传播，并根据需要进行参数更新。
            loss /= self.accumulation_step
            loss.backward()
            # 如果达到累积步数，则执行一次参数更新，并清零优化器的梯度。
            if (i + 1) % self.accumulation_step == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            # 更新进度条上的训练损失信息。
            tbar.set_description("Train loss: %.5f" % (total_losses / (i + 1)))
        # 将当前 epoch 的平均训练损失记录到训练日志中。
        self.log['train']['loss'].append(total_losses / num_batch)

    # 这段代码定义了 valid 方法，用于执行一个 epoch 的验证过程
    def valid(self, epoch):
        # 将模型设置为评估模式，这会关闭 dropout 等训练时的特定操作。
        self.model.eval()
        # 将当前 epoch 的信息添加到验证日志中。
        self.log['valid']['epoch'].append(epoch)
        # 从优化器的状态字典中获取当前学习率，并将其记录到验证日志中。
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        self.log['valid']['lr'].append(lr)
        # 输出当前 epoch 的开始信息，包括 epoch 数、开始时间等。
        start = time.strftime("%H:%M:%S")
        print("Starting epoch: %d | phase: valid | ⏰: %s " % (epoch, start))
        self.log['valid']['time'].append(start)
        # 初始化总验证损失为0，并定义交叉熵损失函数，用于计算模型的输出和真实标签之间的损失。
        total_losses = 0
        criterion = nn.CrossEntropyLoss()
        # 使用 tqdm 创建一个进度条，用于可视化验证进度。
        tbar = tqdm(self.valid_loader, desc="\r")
        num_batch = len(self.valid_loader)
        for i, (log, label) in enumerate(tbar):
            # 遍历验证数据集中的每个批次。
            # 使用 torch.no_grad() 上下文管理器，禁用梯度计算，因为在验证过程中我们不需要进行参数更新。
            with torch.no_grad():
                features = []
                # 将当前批次的日志数据中的每个特征转换为 PyTorch 张量，并移动到指定的计算设备上。
                for value in log.values():
                    features.append(value.clone().detach().to(self.device))
                # 使用模型进行前向传播，得到模型的输出。
                output = self.model(features=features, device=self.device)
                # 计算模型输出与真实标签之间的损失。
                loss = criterion(output, label.to(self.device))
                # 累积当前 epoch 的总验证损失。
                total_losses += float(loss)
        # 更新进度条上的验证损失信息。
        print("Validation loss:", total_losses / num_batch)
        # 将当前 epoch 的平均验证损失记录到验证日志中。
        self.log['valid']['loss'].append(total_losses / num_batch)

        # 如果当前验证损失小于历史最佳验证损失，则更新最佳验证损失，并保存当前模型的检查点。这可以用于在训练过程中选择性地保存模型的最佳版本。
        if total_losses / num_batch < self.best_loss:
            self.best_loss = total_losses / num_batch
            self.save_checkpoint(epoch,
                                 save_optimizer=False,
                                 suffix="bestloss")

    # 这个方法的作用是启动整个训练过程，包括学习率的调整、模型的训练、验证和保存检查点。
    def start_train(self):
        for epoch in range(self.start_epoch, self.max_epoch):
            # 如果当前 epoch 是 0，将学习率除以32。
            # 如果当前 epoch 在 [1, 2, 3, 4, 5] 中，将学习率乘以2。
            # 如果当前 epoch 在 self.lr_step 中，将学习率乘以 self.lr_decay_ratio。
            # 这些操作可以用于调整学习率的策略，以在训练的不同阶段应用不同的学习率。
            if epoch == 0:
                self.optimizer.param_groups[0]['lr'] /= 32
            if epoch in [1, 2, 3, 4, 5]:
                self.optimizer.param_groups[0]['lr'] *= 2
            if epoch in self.lr_step:
                self.optimizer.param_groups[0]['lr'] *= self.lr_decay_ratio
            self.train(epoch)
            # 如果当前 epoch 大于等于总 epoch 数的一半且为偶数时，执行验证操作。
            # 调用 self.valid(epoch) 方法，执行验证操作。
            # 调用 self.save_checkpoint 方法保存检查点，包括模型和可能的优化器状态。
            if epoch >= self.max_epoch // 2 and epoch % 2 == 0:
                self.valid(epoch)
                self.save_checkpoint(epoch,
                                     save_optimizer=True,
                                     suffix="epoch" + str(epoch))
            self.save_checkpoint(epoch, save_optimizer=True, suffix="last")
            self.save_log()
