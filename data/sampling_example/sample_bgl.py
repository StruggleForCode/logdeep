import os
import pandas as pd
import numpy as np
para = {"window_size":0.5,"step_size":0.2,"structured_file":"bgl/BGL_100k_structured.csv","BGL_sequence":'bgl/BGL_sequence.csv'}

def load_BGL():

    structured_file = para["structured_file"]
    # load data
    bgl_structured = pd.read_csv(structured_file) 
    # convert to data time format
    bgl_structured["time"] = pd.to_datetime(bgl_structured["time"],format = "%Y-%m-%d-%H.%M.%S.%f")
    # calculate the time interval since the start time
    bgl_structured["seconds_since"] = (bgl_structured['time']-bgl_structured['time'][0]).dt.total_seconds().astype(int)
    # get the label for each log("-" is normal, else are abnormal label)
    bgl_structured['label'] = (bgl_structured['label'] != '-').astype(int)
    return bgl_structured

# 滑动窗口有点让人不理解
def bgl_sampling(bgl_structured):

    label_data,time_data,event_mapping_data = bgl_structured['label'].values,bgl_structured['seconds_since'].values,bgl_structured['event_id'].values
    log_size = len(label_data)
    # split into sliding window
    start_time = time_data[0]
    start_index = 0
    end_index = 0
    start_end_index_list = []
    # get the first start, end index, end time
    for cur_time in time_data:
        if cur_time < start_time + para["window_size"]*3600:
            end_index += 1
            end_time = cur_time
        else:
            # tuple() 是 Python 内置函数之一，用于将一个可迭代对象转换为元组。
            # 元组是 Python 中的一种序列类型，与列表类似，但是元组是不可变的，即不能修改元组中的元素。
            # 在你的例子中，tuple((start_index,end_index)) 将 (start_index, end_index) 转换为一个元组。
            start_end_pair = tuple((start_index,end_index))
            start_end_index_list.append(start_end_pair)
            break
    # 拿step_size  * 3600 作为窗口， 0.2小时作为窗口，存储数据
    while end_index < log_size:
        start_time = start_time + para["step_size"]*3600
        end_time = end_time + para["step_size"]*3600
        for i in range(start_index,end_index):
            if time_data[i] < start_time:
                i+=1
            else:
                break
        for j in range(end_index, log_size):
            if time_data[j] < end_time:
                j+=1
            else:
                break
        start_index = i
        end_index = j
        start_end_pair = tuple((start_index, end_index))
        start_end_index_list.append(start_end_pair)
    # start_end_index_list is the  window divided by window_size and step_size, 
    # the front is the sequence number of the beginning of the window, 
    # and the end is the sequence number of the end of the window
    inst_number = len(start_end_index_list)
    print('there are %d instances (sliding windows) in this dataset'%inst_number)

    # get all the log indexs in each time window by ranging from start_index to end_index

    expanded_indexes_list=[[] for i in range(inst_number)]
    expanded_event_list=[[] for i in range(inst_number)]

    # event_mapping_data = bgl_structured['event_id'].values
    # 把分好组的 event_id用 expanded_event_list 存储起来
    for i in range(inst_number):
        start_index = start_end_index_list[i][0]
        end_index = start_end_index_list[i][1]
        for l in range(start_index, end_index):
            expanded_indexes_list[i].append(l)
            expanded_event_list[i].append(event_mapping_data[l])
    #=============get labels and event count of each sliding window =========#

    # labels 记录这一组中是否有异常
    labels = []

    # label_data = bgl_structured['label'].values
    for j in range(inst_number):
        label = 0   #0 represent success, 1 represent failure
        for k in expanded_indexes_list[j]:
            # If one of the sequences is abnormal (1), the sequence is marked as abnormal
            if label_data[k]:
                label = 1
                continue
        labels.append(label)
    assert inst_number == len(labels)
    print("Among all instances, %d are anomalies"%sum(labels))

    BGL_sequence = pd.DataFrame(columns=['sequence','label'])
    BGL_sequence['sequence'] = expanded_event_list
    BGL_sequence['label'] = labels
    BGL_sequence.to_csv(para["BGL_sequence"],index=None)

# 总的来说就是分组，对于每一组是否有异常进行检测。labels存储的就是这一组有异常。
if __name__ == "__main__":
    bgl_structured = load_BGL()
    bgl_sampling(bgl_structured)
