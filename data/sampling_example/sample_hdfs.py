import os 
import re
import numpy as np 
import pandas as pd
from collections import OrderedDict

def hdfs_sampling(log_file, window='session', window_size=0):
    assert window == 'session', "Only window=session is supported for HDFS dataset."
    print("Loading", log_file)
    struct_log = pd.read_csv(log_file, engine='c',
            na_filter=False, memory_map=True)
    # OrderedDict() 是 Python 内置函数之一，它是 dict 的一个子类，用于创建一个有序字典。
    # 与普通的 dict 不同，OrderedDict 会记住键值对的插入顺序，因此在遍历时会按照插入顺序返回键值对。
    data_dict = OrderedDict()
    for idx, row in struct_log.iterrows():
        blkId_list = re.findall(r'(blk_-?\d+)', row['Content'])
        blkId_set = set(blkId_list)
        for blk_Id in blkId_set:
            if not blk_Id in data_dict:
                data_dict[blk_Id] = []
            data_dict[blk_Id].append(row['EventId'])
    data_df = pd.DataFrame(list(data_dict.items()), columns=['BlockId', 'EventSequence'])
    data_df.to_csv("hdfs/HDFS_sequence.csv",index=None)

hdfs_sampling('hdfs/HDFS_100k.log_structured.csv')