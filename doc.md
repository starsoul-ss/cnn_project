train 1
Epoch [25/25]: Train Loss: 0.0373, Validation Loss: 1.5197, Validation Accuracy: 0.7982
过拟合

train 2
Epoch [6/25]: Train Loss: 0.0942, Validation Loss: 0.7870, Validation Accuracy: 0.8239
触发早停
测试结果：
Test Loss: 0.7582, Test Accuracy: 0.8174
Confusion Matrix:
[[436   4   9   6  22  71]
 [  9 514   1   7   0  19]
 [ 13   3 449  60  65   9]
 [  7   3  72 451  56   5]
 [  7   0  31  30 475  11]
 [ 79   7   5   3   8 460]]
Classification Report:
              precision    recall  f1-score   support

           0       0.79      0.80      0.79       548
           1       0.97      0.93      0.95       550
           2       0.79      0.75      0.77       599
           3       0.81      0.76      0.78       594
           4       0.76      0.86      0.81       554
           5       0.80      0.82      0.81       562

    accuracy                           0.82      3407
   macro avg       0.82      0.82      0.82      3407
weighted avg       0.82      0.82      0.82      3407

train 3
Epoch [3/25]: Train Loss: 0.7559, Validation Loss: 0.7788, Validation Accuracy: 0.7172
lr:0.001改成了0.005，可以看到损失过大，欠拟合

train 4
Epoch [4/4]: Train Loss: 0.2636, Validation Loss: 0.5098, Validation Accuracy: 0.8302
更改为4个epoch，此处模型的泛化能力相对较好，也没有出现过拟合和欠拟合的情况，测试集准确度提高、损失减小
Test Loss: 0.5410, Test Accuracy: 0.8207
Confusion Matrix:
[[397   8  19  13  28  83]
 [  7 525   2   9   0   7]
 [  5   4 497  60  31   2]
 [  1   2  88 464  39   0]
 [  9   2  46  39 452   6]
 [ 53  15  19   6   8 461]]
Classification Report:
              precision    recall  f1-score   support

           0       0.84      0.72      0.78       548
           1       0.94      0.95      0.95       550
           2       0.74      0.83      0.78       599
           3       0.79      0.78      0.78       594
           4       0.81      0.82      0.81       554
           5       0.82      0.82      0.82       562

    accuracy                           0.82      3407
   macro avg       0.82      0.82      0.82      3407
weighted avg       0.82      0.82      0.82      3407

train 5
Epoch [6/25]: Train Loss: 0.1017, Validation Loss: 1.2559, Validation Accuracy: 0.7671
触发早停
初始lr0.004，变学习率，每次改为之前的0.5倍，学习率有调整，但还是出现过拟合的情况

train 6
Epoch [6/25]: Train Loss: 1.7906, Validation Loss: 1.7923, Validation Accuracy: 0.1739
触发早停
初始lr0.01，变学习率，每次改为0.1倍，有调整，但拟合效果很差

train 7
Epoch [5/25]: Train Loss: 0.0810, Validation Loss: 0.5688, Validation Accuracy: 0.8379
没有按照imagenet的数据集参数进行归一化，模型效果反而更好
Test Loss: 0.5747, Test Accuracy: 0.8397
Confusion Matrix:
[[467   6   6   5  12  52]
 [ 13 525   1   5   1   5]
 [ 17   3 465  61  46   7]
 [  6   1  69 477  39   2]
 [ 18   2  25  37 463   9]
 [ 74  10   7   2   5 464]]
Classification Report:
              precision    recall  f1-score   support

           0       0.78      0.85      0.82       548
           1       0.96      0.95      0.96       550
           2       0.81      0.78      0.79       599
           3       0.81      0.80      0.81       594
           4       0.82      0.84      0.83       554
           5       0.86      0.83      0.84       562

    accuracy                           0.84      3407
   macro avg       0.84      0.84      0.84      3407
weighted avg       0.84      0.84      0.84      3407

train 8
Epoch [15/25]: Train Loss: 0.1840, Validation Loss: 0.4290, Validation Accuracy: 0.8668
采用图片预处理，增加了随机水平翻转与随机旋转正负15度，模型的学习效果更好
Test Loss: 0.4466, Test Accuracy: 0.8579
Confusion Matrix:
[[449   3   8   6  15  67]
 [  5 529   0   6   0  10]
 [ 11   4 482  61  34   7]
 [  4   2  67 494  25   2]
 [  9   3  18  25 490   9]
 [ 58   7   7   3   8 479]]
Classification Report:
              precision    recall  f1-score   support

           0       0.84      0.82      0.83       548
           1       0.97      0.96      0.96       550
           2       0.83      0.80      0.82       599
           3       0.83      0.83      0.83       594
           4       0.86      0.88      0.87       554
           5       0.83      0.85      0.84       562

    accuracy                           0.86      3407
   macro avg       0.86      0.86      0.86      3407
weighted avg       0.86      0.86      0.86      3407

train 9
Epoch [15/25]: Train Loss: 0.3745, Validation Loss: 0.4421, Validation Accuracy: 0.8401
增加dropout，与上一个模型效果相差不大，可能是数据预处理带来的数据增强让额外的正则化不会产生太大影响
Test Loss: 0.4457, Test Accuracy: 0.8453
Confusion Matrix:
[[410  10  11  17  24  76]
 [  3 539   0   8   0   0]
 [  5   5 479  59  45   6]
 [  1   2  72 479  40   0]
 [  2   4  24  32 487   5]
 [ 37  20   4   5  10 486]]
Classification Report:
              precision    recall  f1-score   support

           0       0.90      0.75      0.82       548
           1       0.93      0.98      0.95       550
           2       0.81      0.80      0.81       599
           3       0.80      0.81      0.80       594
           4       0.80      0.88      0.84       554
           5       0.85      0.86      0.86       562

    accuracy                           0.85      3407
   macro avg       0.85      0.85      0.85      3407
weighted avg       0.85      0.85      0.84      3407

train 10
Epoch [21/25]: Train Loss: 0.2307, Validation Loss: 0.4013, Validation Accuracy: 0.8712
将卷积层之后的dropout删除，只保留全连接层之间的dropout，效果有改进
Test Loss: 0.3903, Test Accuracy: 0.8694
Confusion Matrix:
[[459   5   7  10  18  49]
 [  6 534   1   6   0   3]
 [  9   3 488  59  33   7]
 [  4   1  57 500  30   2]
 [  9   0  19  29 490   7]
 [ 51   6   4   4   6 491]]
Classification Report:
              precision    recall  f1-score   support

           0       0.85      0.84      0.85       548
           1       0.97      0.97      0.97       550
           2       0.85      0.81      0.83       599
           3       0.82      0.84      0.83       594
           4       0.85      0.88      0.87       554
           5       0.88      0.87      0.88       562

    accuracy                           0.87      3407
   macro avg       0.87      0.87      0.87      3407
weighted avg       0.87      0.87      0.87      3407

grad-cam进行决策可视化
选择不同分类标签下的hot图，选择了200，395，757，5319，6897，16175六种