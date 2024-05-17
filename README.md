# 说明文档
本仓库为课程《人工智能原理》项目2-图像分类问题
## 代码使用
本工程的结构为
cnn_project/
│
├── data/                  # 存放数据文件夹
│   ├── images/            # 存放图片文件
│   ├── train_label.csv    # 存放图片标签和编号的CSV文件
|   ├── val_label.csv
|   └── test_label.csv
│
├── model.py               # 存放模型定义   
├── model.pth              # 基础模型保存文件
├── model2.pth             # 添加dropout，数据增强后的文件
├── model3.pth             # 少分类问题的模型
├── dataloader.py          # 数据加载和预处理
├── visualization.py       # 可视化工具
├── train.py               # 训练模型的脚本
├── evaluate.py            # 评估模型的脚本
├── doc.md                 # 存放实验数据记录
└── README.md              # 项目的README文件

