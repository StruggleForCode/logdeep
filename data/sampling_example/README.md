# 如何对自己的日志进行采样的示例

* 从 bgl 和 hdfs 数据集中选择前 100k 个日志进行演示. *

## BGL

bgl 数据集仅包含时间信息，因此适用于时间窗

### 1.构建自己的日志

- 读取源码日志
-` 提取标签、时间和原点事件
- 将事件与模板 ID 匹配
  
* BGL中的“-”标签表示正常，否则标签异常。*

`python structure_bgl.py`

### 2.使用滑动窗或固定窗进行采样

通过计算不同日志之间的时间差来使用采样时间窗口。

window_size和step_size的单位是小时。

如果 'step_size=0'，则使用固定窗口;否则，它使用了滑动窗口

“python sample_bgl.py”

## HDFS

bgl 数据集包含block_id信息，因此适合按block_id分组

*block_id代表指定的硬盘存储空间*

### 1.构建自己的日志

与BGL相同...

### 2.使用block_id采样

`python sample_hdfs`