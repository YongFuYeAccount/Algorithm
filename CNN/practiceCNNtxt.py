# -*- coding: utf-8 -*-
#file location the same as spider
# author yyf
""""
固定超参数， 创建pytorch训练数据集，以及测试数据集，
搭建网络模型，训练测试
用的数据集是江老师githup手写字文档txt
cnn输入是四维：总量x 通道数 x 高度 x 宽度
"""""
import torch
import torch.nn as nn
import torchvision   #torchvision里面有一部分是数据集
import matplotlib.pyplot as plt
from os import listdir
import numpy as np
from torch.utils.data import DataLoader

# Hyper Parameters
EPOCH = 5#1           # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 8
LR = 0.001              # learning rate

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


#定义将txt文本中的数据转化为向量的形式
def img2vector(filename):
    returnVect = np.zeros((32, 32))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[i, j] = int(lineStr[j])
    return returnVect

#训练数据
hwLabels = []
trainingFileList = listdir(r'C:\Users\86182\Desktop\LearnText\JHBilibili\Code\Keras基础实战\trainingDigits')
m = len(trainingFileList)
trainingMat = [];
for i in range(m):
    fileNameStr = trainingFileList[i]
    fileStr = fileNameStr.split('.')[0]
    classNumStr = int(fileStr.split('_')[0])
    hwLabels.append(classNumStr)
    trainingMat.append(img2vector(r'C:\Users\86182\Desktop\LearnText\JHBilibili\Code\Keras基础实战\trainingDigits/%s' % fileNameStr))

#将读取的标签和向量数据存到列表中，并reshape他的维度为 总量 x 1 x 32 x 32
trainX=np.array(trainingMat)
trainX=trainX.reshape((trainX.shape[0],1,trainX.shape[1],trainX.shape[2]))

#将nparray 转化为tensor，将其打乱，并重写训练数据集格式为pytorch需要输入的形式
train_x = torch.tensor(trainX).type(torch.FloatTensor)
train_y = torch.tensor(hwLabels)

print(train_x, train_y)


#将标签和输入数据用自定义函数封装
input_data = GetLoader(train_x , train_y)
train_data= DataLoader(input_data, batch_size=BATCH_SIZE, shuffle=True)
print(input_data,train_data)
print(trainX)


#测试集的数据处理
test_hwLabels = []
testFileList = listdir(r'C:\Users\86182\Desktop\LearnText\JHBilibili\Code\Keras基础实战\testDigits')
m = len(testFileList)
testMat = [];
for i in range(m):
    fileNameStr = testFileList[i]
    fileStr = fileNameStr.split('.')[0]
    classNumStr = int(fileStr.split('_')[0])
    test_hwLabels.append(classNumStr)
    testMat.append(img2vector(r'C:\Users\86182\Desktop\LearnText\JHBilibili\Code\Keras基础实战\testDigits/%s' % fileNameStr))

# 打乱索引
#得到的原始数据类型为列表，需要将列表转化为ndarray，在打乱
testMat=np.array(testMat)
test_hwLabels=np.array(test_hwLabels)
# print(type(testMat),type(test_hwLabels))

index = [i for i in range(len(testMat))] # test_data为测试数据
np.random.shuffle(index) # 打乱索引
testMat = testMat[index]
test_hwLabels = test_hwLabels[index]

testX = testMat
testX=testX.reshape((testX.shape[0],1,testX.shape[1],testX.shape[2]))

#将输入数据和标签转化为pytorch要求的tensor格式
test_x = torch.tensor(testX).type(torch.FloatTensor)
test_y = torch.tensor(test_hwLabels)

#建立cnn网络模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 32, 32)维度 图片的宽和高
            nn.Conv2d(
                in_channels=1,              # input 通道数
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 32, 32)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 16, 16)16
        )
        self.conv2 = nn.Sequential(         # input shape (16, 16, 16)
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (16, 16, 16)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 8, 8)
        )
        self.out = nn.Linear(32 * 8 * 8, 10)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)                  #(batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
                                            # x.size(0)表示的是batch
        output = self.out(x)
        return output, x    # return x for visualization


cnn = CNN()
print(cnn)  # net architecture

#优化器和损失函数的选择
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

# training and testing
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_data):   # gives batch data, normalize x when iterate train_loader

        output = cnn(b_x)[0]               # cnn output
        loss = loss_func(output, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients
        if step % 20 == 0:
            test_output, last_layer = cnn(test_x)#拿测试集去验证
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            # print(pred_y,type(pred_y),len(pred_y))#预测的标签有1934个出现的原因主要是因为数据出错了
            # print(test_y.data.numpy(),type(test_y.data.numpy()),len(test_y.data.numpy()))#实际的标签有946个
            # print(float(test_y.size(0)))
            # print(np.sum(pred_y==test_y.data.numpy()))
            # print(float((pred_y == test_y.data.numpy()).astype(int).sum()))
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

# print 10 predictions from test data
test_output, _ = cnn(test_x[:50])
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')
print(test_y[:50].numpy(), 'real number')

















