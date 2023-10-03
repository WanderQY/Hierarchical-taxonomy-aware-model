按10s切割，overlap 5s, minlen 5s。
特征：fbank+一阶+二阶（26*3），在DNN中常用，信息比mfcc多
attention：关注特定频域或时域（个人感觉用空间注意力就足够）
模型：Xception（优点：深度可分离卷积）
数据增强：Mixup
loss: AMSoftmax
