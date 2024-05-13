import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        # 第一层卷积层
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        #self.dropout1 = nn.Dropout(p=0.25)#dropout层，防止过拟合
        
        # 第二层卷积层
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        #self.dropout2 = nn.Dropout(p=0.25)
        
        # 第三层卷积层
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        #self.dropout3 = nn.Dropout(p=0.25)
        
        # 全连接层
        self.fc1 = nn.Linear(64 * 18 * 18, 512)
        self.dropout4 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # 应用第一层卷积层和池化
        x = self.pool(F.relu(self.conv1(x)))
        #x = self.dropout1(x)#激活后应用dropout层
        
        # 应用第二层卷积层和池化
        x = self.pool(F.relu(self.conv2(x)))
        #x = self.dropout2(x)
        
        # 应用第三层卷积层和池化
        x = self.pool(F.relu(self.conv3(x)))
        #x = self.dropout3(x)

        # 展平输出，准备连接全连接层
        x = x.view(-1, 64 * 18 * 18)
        x = F.relu(self.fc1(x))
        x = self.dropout4(x)
        x = self.fc2(x)
        return x
