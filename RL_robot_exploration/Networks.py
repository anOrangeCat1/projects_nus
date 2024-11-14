import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_action):
        super(CNN, self).__init__()

        # 定义网络层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, num_action)
    
    def forward(self, x, keep_rate):
        # 卷积层
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # 展平并经过全连接层
        x = x.view(x.size(0), -1)  # 展平conv3的输出
        x = F.relu(self.fc1(x))

        # 应用dropout
        x = F.dropout(x, p=1 - keep_rate, training=self.training)

        # 输出层
        x = self.fc2(x)
        return x


class LSTM(nn.Module):
    def __init__(self, num_action, num_cell):
        super(LSTM, self).__init__()

        # 定义网络层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(64, 512, kernel_size=7, stride=1)
        self.fc2 = nn.Linear(512, num_action)

        # LSTM层
        self.lstm = nn.LSTM(input_size=num_cell, hidden_size=num_cell, batch_first=True)

    def forward(self, x, keep_rate, trainLength, num_cell,batch_size):
        # 卷积层
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # 展平conv4的输出
        x = x.view(batch_size, trainLength, -1)

        # LSTM层
        h0 = torch.zeros(1, batch_size, num_cell).to(x.device)
        c0 = torch.zeros(1, batch_size, num_cell).to(x.device)
        rnn_out, _ = self.lstm(x, (h0, c0))

        # 展平LSTM输出
        rnn_out = rnn_out.contiguous().view(-1, num_cell)

        # 应用dropout
        rnn_out = F.dropout(rnn_out, p=1 - keep_rate, training=self.training)

        # 输出层
        output = self.fc2(rnn_out)
        return output
    

# 示例用法
# cnn = CNN(num_action=10)
# lstm = LSTM(num_action=10, num_cell=128)

# x = torch.randn(32, 1, 84, 84)  # 示例输入 (batch_size=32, channels=1, 84x84图像)
# keep_rate = 0.5  # Dropout的保留比率
# output = cnn(x, keep_rate)  # 对于CNN
# output = lstm(x, keep_rate, trainLength=10, batch_size=32)  # 对于LSTM