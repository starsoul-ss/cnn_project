# 说明文档
本仓库为课程《人工智能原理》项目2-图像分类问题
## 代码结构
本工程的结构为
```
cnn_project/
│
├── data/                  # 存放数据文件夹
│   ├── images/            # 存放图片文件
│   ├── cam/               # 存放激活图片文件
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

```
## 代码使用
**模型更改**：更改model.py中的语句
**模型训练**：使用train.py,注意更改line14的num_classes为需要更改的分类目标的数量，模型保存在同目录下的model.pth/model2.pth/model3.pth
**模型评估**：使用evalute.py,注意更改line18的num_classes，初始化模型时要导入模型，在line20修改'model.pth'或其他模型
**可视化**：使用visualization.py，注意更改line45的num_classes，在line51的img_path中选择要导入的图片，该图片用于可视化决策过程。在line55中更改第二个参数为图片对应的类别，在label中可以查到
