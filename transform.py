import torch
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader


from torch.utils.tensorboard import SummaryWriter

# 搭建神经网络
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


#加载数据
train_data = torchvision.datasets.CIFAR10("./dataset", train = True, transform = torchvision.transforms.ToTensor(),
                                          download = True)
test_data = torchvision.datasets.CIFAR10("./dataset", train = False, transform = torchvision.transforms.ToTensor(),
                                         download= True)

#长度
train_data_len = len(train_data)
test_data_len = len(test_data)
print("训练数据集的长度为{}".format(train_data_len))
print("测试数据集的长度为{}".format(test_data_len))

#加载数据
train_dataloader = DataLoader(train_data, batch_size = 64)
test_dataloader = DataLoader(test_data, batch_size=64)

#加载神经网络
tudui = Tudui()

#损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
# learning_rate = 0.01
# 1e-2=1 x (10)^(-2) = 1 /100 = 0.01
learning_rate = 1e-2
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10

#搭建tensorboard
writer = SummaryWriter("../log_train")

for i in range(epoch):
    print("------第{}轮训练开始-------".format(i+1))

    #训练步骤开始
    tudui.train()
    for data in train_dataloader:
        imgs, targets = data
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)

        #优化器
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0 :
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

#测试步骤开始

    tudui.eval()
    total_test_loss = 0
    total_acc = 0
    with torch.no_grad():

       for data in test_dataloader:
           imgs, targets = data
           outputs = tudui(imgs)
           loss = loss_fn(outputs, targets)
           total_test_loss = total_test_loss + loss.item()
           acc = (outputs.argmax(1) == targets).sum()
           total_acc = total_acc + acc

    print("整体测试集上的loss：{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_acc/test_data_len))
    writer.add_scalar("total_test_loss", total_test_loss, total_test_step)
    writer.add_scalar("total_acc", total_acc/test_data_len, total_test_step)
    total_test_step = total_test_step + 1
    torch.save(tudui, "tudui_{}.pth".format(i))
    print("模型已保存")

writer.close()










