import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader import create_dataloader
from model import CNN
from dataloader import transform
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device {device}')

# 加载数据
test_loader = create_dataloader('data/test_data.csv', 'data/imgs', batch_size=32, shuffle=False,transform=transform)

# 初始化模型
model = CNN(num_classes=6)
model.load_state_dict(torch.load('model2.pth'))
model.to(device)

# 损失函数
criterion = nn.CrossEntropyLoss()

# 定义评估函数
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predictions = torch.max(outputs, 1)
            total_correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total
    print(f'Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}')
    
    # 计算混淆矩阵和分类报告
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    class_report = classification_report(all_labels, all_predictions, target_names=[str(i) for i in range(6)])

    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)

# 运行评估
evaluate(model, test_loader, criterion, device)
