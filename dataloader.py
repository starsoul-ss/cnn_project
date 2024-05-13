import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None,exclude_labels=None):
        """
        初始化函数
        :param annotations_file: CSV文件路径，包含图像名称和标签
        :param img_dir: 图像存储的目录
        :param transform: 进行的预处理函数
        """
        self.img_labels = pd.read_csv(annotations_file)
        if exclude_labels:
            self.img_labels = self.img_labels[~self.img_labels['label'].isin(exclude_labels)]
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        """
        返回数据集中的图像数量
        """
        return len(self.img_labels)

    def __getitem__(self, idx):
        """
        根据索引号获取数据集中的图像和标签
        :param idx: 图像的索引号
        """
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx]['image_name'])
        image = Image.open(img_path).convert("RGB")  # 确保图像为RGB
        label = self.img_labels.iloc[idx]['label']
        if self.transform:
            image = self.transform(image)
        return image, label

# 定义数据转换
# 注意：可以根据需要添加更多的数据增强
transform = transforms.Compose([
    transforms.Resize((150, 150)),  # 确认所有图像都是150x150
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(15),      # 随机旋转±15度
    transforms.ToTensor(),  # 将图像转换为torch.Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化，使用imagenet的均值和标准差
])

# 创建数据加载器
def create_dataloader(csv_file, img_dir, batch_size=32, shuffle=True, transform=transform):#shuffle=True表示每个epoch都打乱数据
    dataset = CustomImageDataset(annotations_file=csv_file, img_dir=img_dir, transform=transform,exclude_labels=[2,3,4,5])
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
