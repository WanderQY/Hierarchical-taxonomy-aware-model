##coding:utf-8
import matplotlib.pyplot as plt
import numpy as np
import itertools
from matplotlib import rcParams

# 定义混淆矩阵数据
confusion_matrix = np.array(
    [[451, 1, 12, 6, 1],
     [18, 451, 5, 9, 14],
     [41, 27, 487, 2, 15],
     [13, 21, 5, 395, 4],
     [1, 4, 15, 19, 421]])

# 计算每个类别的准确率
class_accuracy = confusion_matrix / confusion_matrix.sum(axis=1)[:, None]

# 绘制混淆矩阵图像
# 要想改变颜色，修改cmap参数，红色：plt.cm.Reds
plt.imshow(class_accuracy, cmap=plt.cm.Blues)

# 添加网格
plt.grid(False)
plt.colorbar()
# labels表示你不同类别的代号，这里有5个类别
labels = ['0', '1', '2', '3', '4']
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize=12)
plt.yticks(tick_marks, labels, fontsize=12)

# 在每个小格子中显示相应的数值和准确率
for i, j in itertools.product(range(class_accuracy.shape[0]), range(class_accuracy.shape[1])):
    # 显示数值
    value = confusion_matrix[i, j]
    plt.text(j, i, value, horizontalalignment="center", color="white" if class_accuracy[i, j] > 0.5 else "black")

    # 显示准确率
    acc = class_accuracy[i, j] * 100
    plt.text(j, i + 0.3, f"{acc:.2f}%", horizontalalignment="center", color="black")

# 添加x和y轴标签
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.title("Confusion Matrix")

