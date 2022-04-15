# -*- coding: utf-8 -*-
#file location the same as spider
# author yyf
""""
用DNN分类实验室csv数据，训练集和测试集都用load形式
数据集是实验室数据集
accuracy = 93%
70%train
30%test
"""""
import torch
import torch.nn as nn
import csv
import numpy as np
from torch.utils.data import DataLoader


torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 30              # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 32        #批次训练的数据
LR = 0.001

#构建数据集函数
# 定义GetLoader类，继承Dataset方法，并重写__getitem__()和__len__()方法
class GetLoader(torch.utils.data.Dataset):
	# 初始化函数，得到数据
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label
    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels
    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)

#读csv函数(读数据)
def R_xcsv(Filedirectory):
    csv_file = open(Filedirectory, encoding='utf-8')  # 打开csv文件
    csv_reader_lines = csv.reader(csv_file)  # 逐行读取csv文件
    date = []  # 创建列表准备接收csv各行数据
    renshu = 0
    for one_line in csv_reader_lines:
        date.append(one_line)  # 将读取的csv分行数据按行存入列表‘date’中
        renshu = renshu + 1  # 统计行数

    #读取的date是字符串，将其转化为数值
    for i in range(len(date)):
        date[i] = list(map(float, date[i]))
    # for i in range(len(date)):
    #     date[i] = np.array(date[i]).reshape(8, 8)#将列表的元素转化为8 x 8
    # date = np.array(date, dtype=float)  # trainX为待转化的列表
    return date


#读标签函数
def R_ycsv(Filedirectory):
    csv_file = open(Filedirectory, encoding='utf-8')  # 打开csv文件
    csv_reader_lines = csv.reader(csv_file)  # 逐行读取csv文件
    date = []  # 创建列表准备接收csv各行数据
    renshu = 0
    for one_line in csv_reader_lines:
        date.append(one_line)  # 将读取的csv分行数据按行存入列表‘date’中
        renshu = renshu + 1  # 统计行数

    for i in range(len(date)):#将读取的字符串转化为数值
        date[i] = list(map(int, date[i]))
    # date = np.array(date, dtype=float)  # trainX为待转化的列表
    return date

#构建pytorch行驶本数据集函数
# 定义GetLoader类，继承Dataset方法，并重写__getitem__()和__len__()方法
class GetLoader(torch.utils.data.Dataset):
	# 初始化函数，得到数据
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label
    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels
    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)

#训练数据的构建
trainX1=R_xcsv(r'C:\Users\86182\Desktop\Dataset\data\d0.csv')
trainX2=R_xcsv(r'C:\Users\86182\Desktop\Dataset\data\d1.csv')
trainX3=R_xcsv(r'C:\Users\86182\Desktop\Dataset\data\d2.csv')
trainX4=R_xcsv(r'C:\Users\86182\Desktop\Dataset\data\d3.csv')
trainX5=R_xcsv(r'C:\Users\86182\Desktop\Dataset\data\d4.csv')
trainingMat=np.concatenate((trainX1,trainX2,trainX3,trainX4,trainX5),axis=0)#垂直组合numpy
# trainingMat=np.concatenate((trainX1,trainX2,trainX3),axis=0)#垂直组合numpy

hwLabels1=R_ycsv(r'C:\Users\86182\Desktop\Dataset\label\d0.csv')
hwLabels2=R_ycsv(r'C:\Users\86182\Desktop\Dataset\label\d1.csv')
hwLabels3=R_ycsv(r'C:\Users\86182\Desktop\Dataset\label\d2.csv')
hwLabels4=R_ycsv(r'C:\Users\86182\Desktop\Dataset\label\d3.csv')
hwLabels5=R_ycsv(r'C:\Users\86182\Desktop\Dataset\label\d4.csv')
train_hwLabels=np.concatenate((hwLabels1,hwLabels2,hwLabels3,hwLabels4,hwLabels5),axis=0)#垂直组合numpy
# train_hwLabels = train_hwLabels.reshape(len(train_hwLabels),) # 修改成一维
# train_hwLabels=np.concatenate((hwLabels1,hwLabels2,hwLabels3),axis=0)#垂直组合numpy



# 打乱索引
#得到的原始数据类型为列表，需要将列表转化为ndarray，在打乱
trainingMat=np.array(trainingMat)
train_hwLabels=np.array(train_hwLabels).reshape(len(train_hwLabels),) # 将标签修改成一维


index = [i for i in range(len(trainingMat))] # test_data为测试数据
np.random.shuffle(index) # 打乱索引
trainingMat = trainingMat[index]
train_hwLabels = train_hwLabels[index]


# #将读取的标签和向量数据存到列表中，并reshape他的维度为 总量  x 8 x 8
totaldata =np.array(trainingMat)
totaldata = totaldata.reshape((len(totaldata),8,8))
totallable = train_hwLabels

# 划分训练集和测试集数据
trainX = totaldata[0:151200]
train_Labels = totallable[0:151200]

testX = totaldata[151201:]
test_Labels = totallable[151201:]

#将nparray 转化为tensor，将其打乱，并重写训练数据集格式为pytorch需要输入的形式
train_x = torch.tensor(trainX).type(torch.FloatTensor)
train_y = torch.tensor(train_hwLabels, dtype=torch.long)#csv读取int转化为long


 #将标签和输入数据用自定义函数封装
input_traindata = GetLoader(train_x , train_y)
train_data= DataLoader(input_traindata, batch_size=BATCH_SIZE, shuffle=True)


# #划分测试集总的数据为216000

test_x = torch.tensor(testX).type(torch.FloatTensor)
test_y = torch.tensor(test_Labels, dtype=torch.long)#csv读取int转化为long

#将标签和输入数据用自定义函数封装
input_testdata = GetLoader(train_x , train_y)
test_data= DataLoader(input_testdata, batch_size=BATCH_SIZE, shuffle=True)


#搭建模型
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(64, 50),#输入是8 x 8 =64,输入的特征是64，隐含层50 ；in_features=64, out_features=50, bias=True)
            nn.LeakyReLU(inplace=True),
            nn.Linear(50, 32),#in_features=50, out_features=32
            nn.LeakyReLU(inplace=True),
            nn.Linear(32, 5),#输出5个特征 ；in_features=200, out_features=10, bias=True)
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.model(x)
        return x
    #[-3.6963e-02,  3.2585e-01,  8.7656e+00,  2.3879e+00, -6.7743e-02,-1.1572e-03, -2.8706e-02, -1.9428e-02,  1.4496e+00, -3.1113e-02]]
     #返回的 X 是上述的一个张量，各种类别的概率


dnn = DNN()
print(dnn)

# 优化器和损失函数的选择
optimizer = torch.optim.Adam(dnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

#训练过程
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_data):   # gives batch data, normalize x when iterate train_loader
                                                    #b_x一个批次里面的所有输入数据组成的张量 ,b_y该batchsize里面的数据的标签张量
        b_x = b_x.view(-1, 8 * 8)  # reshape x to (batch, input_feature)转化为二维张量，batchsize X （输入数据的尺寸（8*8））
        output = dnn(b_x)  # dnn output返回的是这个批次里面每个输入数据，在各个类别下的概率的张量
        loss = loss_func(output, b_y)  # cross entropy loss

        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        if step % 100 == 0:#每一百步
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.
                  format(epoch, step * len( b_x), len(train_data.dataset), 100. * step / len(train_data),
                         loss.item()))

    test_loss = 0
    correct = 0
    for data_x, data_y in test_data:
        data_x = data_x.view(-1,8 * 8)  # batch x 输入数据28*28
        pred_y= dnn(data_x)
        test_loss += loss_func(pred_y, data_y).item()

        pred = pred_y.data.max(1)[1]
        correct += pred.eq(data_y.data).sum()

    test_loss /= len(test_data.dataset)
    print('\nTest set: Average loss: {:.4f},Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_data.dataset), 100. * correct / len(test_data.dataset)))

