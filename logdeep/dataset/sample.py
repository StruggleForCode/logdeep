import json
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm


def read_json(filename):
    with open(filename, 'r') as load_f:
        file_dict = json.load(load_f)
    return file_dict


def trp(l, n):
    """ Truncate or pad a list """
    r = l[:n]
    if len(r) < n:
        r.extend(list([0]) * (n - len(r)))
    return r


def down_sample(logs, labels, sample_ratio):
    print('sampling...')
    total_num = len(labels)
    all_index = list(range(total_num))
    sample_logs = {}
    for key in logs.keys():
        sample_logs[key] = []
    sample_labels = []
    sample_num = int(total_num * sample_ratio)

    for i in tqdm(range(sample_num)):
        random_index = int(np.random.uniform(0, len(all_index)))
        for key in logs.keys():
            sample_logs[key].append(logs[key][random_index])
        sample_labels.append(labels[random_index])
        del all_index[random_index]
    return sample_logs, sample_labels

'''
这段代码定义了一个名为`sliding_window`的函数，它用于从HDFS（Hadoop Distributed File System）日志数据中生成滑动窗口的数据集。以下是对代码的详细解释：

1. **函数签名**：
    - `sliding_window(data_dir, datatype, window_size, sample_ratio=1)`
    - 参数：
        - `data_dir`: 包含HDFS日志数据的目录路径。
        - `datatype`: 数据集的类型，可以是'train'或'val'。
        - `window_size`: 滑动窗口的大小。
        - `sample_ratio`: 数据集的采样比例，默认为1。

2. **数据结构初始化**：
    - `event2semantic_vec`: 从JSON文件中读取的事件到语义向量的映射。
    - `num_sessions`: 记录会话的数量。
    - `result_logs`: 一个字典，包含三个键：'Sequentials'、'Quantitatives'和'Semantics'，分别对应序列、数量和语义的日志信息。
    - `labels`: 用于存储标签的列表。

3. **根据数据集类型设置数据目录**：
    - 如果`datatype`为'train'，则`data_dir`追加为'hdfs/hdfs_train'。
    - 如果`datatype`为'val'，则`data_dir`追加为'hdfs/hdfs_test_normal'。

4. **打开数据文件**：
    - 使用`open`函数打开指定路径的数据文件。
    - 对文件的每一行进行处理。

5. **滑动窗口处理**：
    - 使用`line.strip().split()`将每一行的数字字符串分割成一个整数元组。
    - 使用`map`和`lambda`函数将元组中的每个数字减1。
    - 对于每个会话，通过滑动窗口提取子序列，并为每个子序列生成三种模式：Sequential、Quantitative和Semantic。
        - `Sequential_pattern`: 滑动窗口内的整数子序列。
        - `Quantitative_pattern`: 滑动窗口内每个值的计数。
        - `Semantic_pattern`: 滑动窗口内每个事件对应的语义向量。

6. **数据格式处理**：
    - 将处理后的子序列、数量和语义模式添加到`result_logs`字典中。
    - 将滑动窗口的下一个值作为标签，并添加到`labels`列表中。

7. **数据集采样**：
    - 如果`sample_ratio`不等于1，则使用`down_sample`函数对数据集进行下采样。

8. **输出信息**：
    - 打印数据集的信息，包括文件路径、会话数量和序列数量。

9. **返回结果**：
    - 返回`result_logs`和`labels`作为函数的输出。
'''

def sliding_window(data_dir, datatype, window_size, sample_ratio=1):
    '''
    dataset structure
        result_logs(dict):
            result_logs['feature0'] = list()
            result_logs['feature1'] = list()
            ...
        labels(list)
    '''
    # 从JSON文件中读取事件到语义向量的映射
    event2semantic_vec = read_json(data_dir + 'hdfs/event2semantic_vec.json')
    # 记录会话的数量
    num_sessions = 0
    # 初始化存储日志信息和标签的数据结构
    result_logs = {}
    result_logs['Sequentials'] = []
    result_logs['Quantitatives'] = []
    result_logs['Semantics'] = []
    labels = []
    # 如果是训练集，将数据目录路径追加为'hdfs/hdfs_train'
    if datatype == 'train':
        data_dir += 'hdfs/hdfs_train'
    # 如果是验证集，将数据目录路径追加为'hdfs/hdfs_test_normal'
    if datatype == 'val':
        data_dir += 'hdfs/hdfs_test_normal'

    # 打开数据文件
    with open(data_dir, 'r') as f:
        # 对文件的每一行进行处理
        for line in f.readlines():
            # 记录会话数量
            num_sessions += 1
            # 这段代码的作用是将输入的字符串（可能包含空白字符分隔的整数）转换为一个元组，其中每个整数都减去1。
            # 例如，如果line是字符串"1 2 3"，则经过处理后的元组将是(0, 1, 2)。
            # line.strip().split(): 首先，line.strip()会移除字符串两端的空白字符（如空格、制表符、换行符等），然后split()函数将其分割成一个由空白字符分隔的字符串列表
            # map(int, ...): 对上一步得到的每个字符串元素应用int函数，将其转换为整数
            # lambda n: n - 1: 定义了一个匿名函数（lambda函数），该函数接受一个参数n，并返回n - 1的结果
            # map(lambda ..., map(...)): 使用map函数，将上述的匿名函数应用于第2步得到的整数列表中的每个元素，得到一个新的由这些处理过的元素组成的列表。
            # tuple(...): 最后，使用tuple()函数将处理过的列表转换为元组。
            line = tuple(map(lambda n: n - 1, map(int, line.strip().split())))

            # 对于每个会话，通过滑动窗口提取子序列
            for i in range(len(line) - window_size):
                # 生成Sequential模式，即滑动窗口内的整数子序列
                Sequential_pattern = list(line[i:i + window_size])
                # 生成Quantitative模式，即滑动窗口内每个值的计数
                Quantitative_pattern = [0] * 28
                # 对每一个取值进行统计
                log_counter = Counter(Sequential_pattern)

                for key in log_counter:
                    Quantitative_pattern[key] = log_counter[key]
                # 生成Semantic模式，即滑动窗口内每个事件对应的语义向量
                Semantic_pattern = []
                for event in Sequential_pattern:
                    if event == 0:
                        Semantic_pattern.append([-1] * 300)
                    else:
                        Semantic_pattern.append(event2semantic_vec[str(event - 1)])

                # 转换为numpy数组并处理维度
                Sequential_pattern = np.array(Sequential_pattern)[:, np.newaxis]
                Quantitative_pattern = np.array(Quantitative_pattern)[:, np.newaxis]

                # 将模式添加到result_logs字典中
                result_logs['Sequentials'].append(Sequential_pattern)
                result_logs['Quantitatives'].append(Quantitative_pattern)
                result_logs['Semantics'].append(Semantic_pattern)
                # 将滑动窗口的下一个值作为标签，并添加到labels列表中
                labels.append(line[i + window_size])

    # 如果采样比例不等于1，进行下采样
    if sample_ratio != 1:
        result_logs, labels = down_sample(result_logs, labels, sample_ratio)

    # 打印文件路径、会话数量和序列数量的信息
    print('File {}, number of sessions {}'.format(data_dir, num_sessions))
    print('File {}, number of seqs {}'.format(data_dir,
                                              len(result_logs['Sequentials'])))
    # 返回处理后的数据和标签
    return result_logs, labels


def session_window(data_dir, datatype, sample_ratio=1):
    event2semantic_vec = read_json(data_dir + 'hdfs/event2semantic_vec.json')
    result_logs = {}
    result_logs['Sequentials'] = []
    result_logs['Quantitatives'] = []
    result_logs['Semantics'] = []
    labels = []

    if datatype == 'train':
        data_dir += 'hdfs/robust_log_train.csv'
    elif datatype == 'val':
        data_dir += 'hdfs/robust_log_valid.csv'
    elif datatype == 'test':
        data_dir += 'hdfs/robust_log_test.csv'

    train_df = pd.read_csv(data_dir)
    for i in tqdm(range(len(train_df))):
        ori_seq = [
            int(eventid) for eventid in train_df["Sequence"][i].split(' ')
        ]
        Sequential_pattern = trp(ori_seq, 50)
        Semantic_pattern = []
        for event in Sequential_pattern:
            if event == 0:
                Semantic_pattern.append([-1] * 300)
            else:
                Semantic_pattern.append(event2semantic_vec[str(event - 1)])
        Quantitative_pattern = [0] * 29
        log_counter = Counter(Sequential_pattern)

        for key in log_counter:
            Quantitative_pattern[key] = log_counter[key]

        Sequential_pattern = np.array(Sequential_pattern)[:, np.newaxis]
        Quantitative_pattern = np.array(Quantitative_pattern)[:, np.newaxis]
        result_logs['Sequentials'].append(Sequential_pattern)
        result_logs['Quantitatives'].append(Quantitative_pattern)
        result_logs['Semantics'].append(Semantic_pattern)
        labels.append(int(train_df["label"][i]))

    if sample_ratio != 1:
        result_logs, labels = down_sample(result_logs, labels, sample_ratio)

    # result_logs, labels = up_sample(result_logs, labels)

    print('Number of sessions({}): {}'.format(data_dir,
                                              len(result_logs['Semantics'])))
    return result_logs, labels
