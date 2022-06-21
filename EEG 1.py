# 导入包
import torch

torch.set_default_tensor_type(torch.FloatTensor)
import numpy as np
import scipy.io
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F


# 定义网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=16,
                out_channels=16,
                kernel_size=3,  # 卷积核3*3
                stride=1,  # 步长
                padding=1  # 边缘填充
            ),  # 维度变换(20,16,840) --> (20,16,840)
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)  # 维度变换(20,16,840) --> (20,16,420)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),  # 维度变换(20,16,420) --> (20,32,420)
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)  # 维度变换(20,32,420) --> (20,32,210)
        )
        self.output = nn.Linear(32 * 210, 2)  # 这里的32*210是数字，不代表二维32维度乘以210，训练数据有变化的时候，这里必须要改

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        # out.view平铺到一维，类似于keras中的flatten
        out = out.view(out.size(0), -1)  # pytorch的展平操作是在forward中定义的，在这里维度变为二维20*（32*210）
        out = self.output(out)
        return out


cnn = CNN()
print(cnn)  # 打印查看网络结构

tbase = 0.2  # 基线长度
ttrial = 0.5  # 关注的脑电时间段
fs = 1200  # 采样率
# 准备数据
# make data
## load eegdata
# samplesfile = scipy.io.loadmat('/Users/thrive/Documents/Dong/research/mypaper/code/dataset/rawdata/subject1.mat') #读入字典
samplesfile = scipy.io.loadmat('./dataset/sample.mat')  # 读入字典
print(samplesfile.shape)
eegdata = samplesfile['eeg']  # 提取numpy数组,结构体名称是eeg，包含字段有name date data1 marker1 data2 marker2……
#eegdata1 = samplesfile['X']
#eegdata = np.reshape(eegdata1,(20, 16, int((tbase + ttrial) * fs)))
#eegdata = np.random.uniform(0, 1, size = (20, 16, int((tbase + ttrial) * fs)))
#eegdata = eegdata2.astype(np.int)
print('# test eegdata:', eegdata[0, 0]['marker1'][:2, :1], '\n')  # 打印出来marker1字段看一下
#四维的脑电数据
tbase = 0.2  # 基线长度
ttrial = 0.5  # 关注的脑电时间段
fs = 1200  # 采样率

## train data
### epoch data将数据分段
epochdata = np.zeros((20, 16, int((tbase + ttrial) * fs)))  # 初始化一个数组用于存放分段后的脑电数据
#20指的是20个事件（marker）,16指16个通道，int((tbase + ttrial) * fs))指脑电数据event_time

i = 0
for temp in epochdata:
    epochdata[i, :, :] = eegdata[0, 0]['data1'][:, round((eegdata[0, 0]['marker1'][i, 1] - tbase) * fs):round(
        (eegdata[0, 0]['marker1'][i, 1] - tbase) * fs) + int((tbase + ttrial) * fs)]
    #这里将data1和maker1全部赋值个epochdate中的20个事件
    i += 1
epochdata = torch.FloatTensor(epochdata)  # 将分段后的数据转化为float类型，否则无法进行训练
print('# test shape of 3D data:', epochdata.shape, '\n')
print('# test epoched data:', epochdata[:2, 2:3, :2], '\n')

### make labels做标签
labels = np.zeros((20))  # 初始化标签数组
labels = torch.LongTensor(eegdata[0, 0]['marker1'][:, 2])  # 标签需要转化为long类型，否则无法进行训练
print('# test labels shape:', labels.shape, '\n')
print('# test labels:', labels[0])

### make train data
batch_size = 20
traindataset = Data.TensorDataset(epochdata, labels)
train_loader = Data.DataLoader(traindataset, batch_size, shuffle=True)  # 随机读取小批量
for X, y in train_loader:
    print(X, y)
    break

## test data 按照上边的方法准备测试集，但最后有些不同
epochdata = np.zeros((20, 16, int((tbase + ttrial) * fs)))

i = 0
for temp in epochdata:
    epochdata[i, :, :] = eegdata[0, 0]['data2'][:, round((eegdata[0, 0]['marker2'][i, 1] - tbase) * fs):round(
        (eegdata[0, 0]['marker2'][i, 1] - tbase) * fs) + int((tbase + ttrial) * fs)]
    i += 1
epochdata = torch.FloatTensor(epochdata)
print('# test shape of 3D data:', epochdata.shape, '\n')
print('# test epoched data:', epochdata[:2, 2:3, :2], '\n')

### make labels
labels = np.zeros((20))
labels = torch.LongTensor(eegdata[0, 0]['marker2'][:, 2])
print('# test labels shape:', labels.shape, '\n')
print('# test labels:', labels[0])
# 训练集中会自动进行归一化，测试集要手动归一化一哈(应该是在下边的代码中针对训练集和测试集的区别，所以需要这一步)
m = nn.LayerNorm([20, 16, 840])
test_x = m(epochdata)
test_y = labels

# 准备训练
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001, )  # 使用adam优化算法进行训练
loss_func = nn.CrossEntropyLoss()  # 损失函数选择交叉熵损失函数
for epoch in range(30):
    for step, (b_x, b_y) in enumerate(train_loader):
        output = cnn(b_x)
        # b_y必须为long型
        loss = loss_func(output, b_y)

        optimizer.zero_grad()  # 梯度置零
        loss.backward()
        optimizer.step()  # 参数更新

        if step % 20 == 0:
            test_output = cnn(test_x)
            # .max(a,1)返回a中每行的最大值,[1].data.numpy(),返回最大值的位置索引
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
print('finish training')
torch.save(cnn, 'cnn_minist.pkl')  # 保存训练好的网络

# 使用训练好的网络预测
cnn = torch.load('cnn_minist.pkl')

test_output = cnn(test_x[:20])
pred_y = torch.max(test_output, 1)[1].data.numpy()

print(pred_y, 'prediction number')
print(test_y[:20].numpy(), 'real number')

test_output1 = cnn(test_x)
pred_y1 = torch.max(test_output1, 1)[1].data.numpy()
accuracy = float((pred_y1 == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
print('accuracy', accuracy)
