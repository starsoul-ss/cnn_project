train 
Epoch [25/25]: Train Loss: 0.0373, Validation Loss: 1.5197, Validation Accuracy: 0.7982

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
