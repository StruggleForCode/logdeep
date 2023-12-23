import torch
import torch.nn as nn
from torch.autograd import Variable


# 它是一个基于长短时记忆网络（LSTM）的模型，用于处理序列数
# 这个模型适用于处理序列数据，其中LSTM层用于捕捉序列中的长期依赖关系，而全连接层则将最终的隐藏状态映射到输出空间。
# 这样的模型可以用于各种任务，例如时间序列预测、日志异常检测等。
class deeplog(nn.Module):
    # 初始化模型的结构。
    # 接受四个参数：input_size 表示输入序列的特征维度，hidden_size 表示LSTM隐藏状态的维度，num_layers 表示LSTM层数，num_keys 表示模型最终输出的维度。
    # 创建了一个nn.LSTM层，用于处理输入序列，以及一个全连接层nn.Linear，用于生成最终的输出。
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(deeplog, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # 定义LSTM层
        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            num_layers,
                            batch_first=True)
        # 定义全连接层
        self.fc = nn.Linear(hidden_size, num_keys)

    # 定义了模型的前向传播过程。接受两个参数：features 是输入的序列数据，device 是设备信息。
    # 从输入序列中取出第一个特征作为初始输入。
    # 初始化LSTM的隐藏状态和细胞状态为零。
    # 将输入序列传递给LSTM层，获取LSTM的输出。
    # 从LSTM输出中取出最后一个时间步的隐藏状态，并通过全连接层得到最终输出。
    def forward(self, features, device):
        # 获取输入序列的第一个特征
        input0 = features[0]
        # 初始化LSTM的隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, input0.size(0),
                         self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, input0.size(0),
                         self.hidden_size).to(device)
        # 将输入序列传递给LSTM层
        out, _ = self.lstm(input0, (h0, c0))
        # 从LSTM输出中取出最后一个时间步的隐藏状态，并通过全连接层得到最终输出
        out = self.fc(out[:, -1, :])
        return out


class loganomaly(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(loganomaly, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm0 = nn.LSTM(input_size,
                             hidden_size,
                             num_layers,
                             batch_first=True)
        self.lstm1 = nn.LSTM(input_size,
                             hidden_size,
                             num_layers,
                             batch_first=True)
        self.fc = nn.Linear(2 * hidden_size, num_keys)
        self.attention_size = self.hidden_size

        self.w_omega = Variable(
            torch.zeros(self.hidden_size, self.attention_size))
        self.u_omega = Variable(torch.zeros(self.attention_size))

        self.sequence_length = 28

    def attention_net(self, lstm_output):
        output_reshape = torch.Tensor.reshape(lstm_output,
                                              [-1, self.hidden_size])
        attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega))
        attn_hidden_layer = torch.mm(
            attn_tanh, torch.Tensor.reshape(self.u_omega, [-1, 1]))
        exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer),
                                    [-1, self.sequence_length])
        alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1])
        alphas_reshape = torch.Tensor.reshape(alphas,
                                              [-1, self.sequence_length, 1])
        state = lstm_output
        attn_output = torch.sum(state * alphas_reshape, 1)
        return attn_output

    def forward(self, features, device):
        input0, input1 = features[0], features[1]

        h0_0 = torch.zeros(self.num_layers, input0.size(0),
                           self.hidden_size).to(device)
        c0_0 = torch.zeros(self.num_layers, input0.size(0),
                           self.hidden_size).to(device)

        out0, _ = self.lstm0(input0, (h0_0, c0_0))

        h0_1 = torch.zeros(self.num_layers, input1.size(0),
                           self.hidden_size).to(device)
        c0_1 = torch.zeros(self.num_layers, input1.size(0),
                           self.hidden_size).to(device)

        out1, _ = self.lstm1(input1, (h0_1, c0_1))
        multi_out = torch.cat((out0[:, -1, :], out1[:, -1, :]), -1)
        out = self.fc(multi_out)
        return out


class robustlog(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(robustlog, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, num_keys)

    def forward(self, features, device):
        input0 = features[0]
        h0 = torch.zeros(self.num_layers, input0.size(0),
                         self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, input0.size(0),
                         self.hidden_size).to(device)
        out, _ = self.lstm(input0, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
