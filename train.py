import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import create_dataloader
from dataloader import transform
from model import CNN

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 配置参数
num_classes = 6
learning_rate = 0.001
batch_size = 32
num_epochs = 25

# 加载数据
train_loader = create_dataloader('data/train_data.csv', 'data/imgs', batch_size=batch_size, shuffle=True,transform=transform)
val_loader = create_dataloader('data/val_data.csv', 'data/imgs', batch_size=batch_size, shuffle=False,transform=transform)

# 初始化模型
model = CNN(num_classes).to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 定义验证函数
def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total_loss += loss.item()
            total_correct += (predicted == labels).sum().item()
    avg_loss = total_loss / len(val_loader)
    accuracy = total_correct / len(val_loader.dataset)
    return avg_loss, accuracy

# 训练模型
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        # 在每个epoch后进行验证
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        print(f'Epoch [{epoch+1}/{num_epochs}]: Train Loss: {total_train_loss / len(train_loader):.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

# 开始训练和验证
train(model, train_loader, val_loader, criterion, optimizer, num_epochs)

# 保存模型
torch.save(model.state_dict(), 'model.pth')
print('Model saved to model.pth')
