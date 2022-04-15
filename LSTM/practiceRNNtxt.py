# -*- coding: utf-8 -*-
#file location the same as spider
# author yyf

""""
用LSTM分类文档类型的txt数据
数据集是江老师githup手写字的txt数据集
lstm输入是三维：总量x 长 x 高（行的长度 x 列的数量）
"""""
import torch
import torch.nn as nn
import torchvision   #torchvision里面有一部分是数据集
import matplotlib.pyplot as plt
from os import listdir
import numpy as np
from torch.utils.data import DataLoader


torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 10               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 32        #批次训练的数据
TIME_STEP = 32       #28   rnn time step / image height 考虑多少个时间点的数据，每
                        # 个时间点上面给rnn多少个数据点，每28步读取一行信息
INPUT_SIZE = 32       #28    rnn input size / image width
LR = 0.01
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


# #将读取的标签和向量数据存到列表中，并reshape他的维度为 总量 x 1 x 32 x 32
# trainX=np.array(trainingMat)
# trainX=trainX.reshape((trainX.shape[0],1,trainX.shape[1],trainX.shape[2]))


trainX=np.array(trainingMat)
#将nparray 转化为tensor，将其打乱，并重写训练数据集格式为pytorch需要输入的形式
train_x = torch.tensor(trainX).type(torch.FloatTensor)
train_y = torch.tensor(hwLabels)

#将标签和输入数据用自定义函数封装
input_data = GetLoader(train_x , train_y)
train_data= DataLoader(input_data, batch_size=BATCH_SIZE, shuffle=True)



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


# #将测试集数据转化为四维
# testX = testMat
# testX=testX.reshape((testX.shape[0],1,testX.shape[1],testX.shape[2]))

testX = testMat
#将输入数据和标签转化为pytorch要求的tensor格式
test_x = torch.tensor(testX).type(torch.FloatTensor)
test_y = torch.tensor(test_hwLabels)

# print(train_data)
# print(test_x,test_y)
# print(type(test_x),type(test_y))

#搭建rnn模型
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()#继承module里面的属性

        self.rnn = nn.LSTM(         # if use nn.RNN()直接使用rnn准确率不高, it hardly learns直接使用lstm模型
            input_size=INPUT_SIZE,  #每个时间段input28个pixle点
            hidden_size=64,         # rnn hidden unit
            num_layers=2,           # number of rnn layer 大一点可能精度高一点，但是需要算力
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )                           #输入数据的维度batch_size在第一个维度就是true，第二个该项就是false

        self.out = nn.Linear(64, 10)  #rnn输出的数据，64个隐含层的元素，10个类别

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state

        # choose r_out at the last time step选取最后一个时间点进行判断，因为要读完所有数据之后在做处理
        out = self.out(r_out[:, -1, :])
        return out

rnn = RNN()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()    #内部转化为标签分类的结果是什么就是什么不是(0,1,0,0,)的形式，显示的不是独热编码
                                        # the target label is not one-hotted

# training and testing
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_data):        # gives batch data
        b_x = b_x.view(-1, 32, 32)              # reshape x to (batch, time_step, input_size)
                                                #rnn接收数据的形式

        output = rnn(b_x)                               # rnn output
        loss = loss_func(output, b_y)                   # cross entropy loss
        optimizer.zero_grad()                           # clear gradients for this training step
        loss.backward()                                 # backpropagation, compute gradients
        optimizer.step()                                # apply gradients

        if step % 10 == 0:
            test_output = rnn(test_x)                   # (samples, time_step, input_size)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            # accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)


# print 10 predictions from test data
test_output = rnn(test_x[:10].view(-1, 32, 32))
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')
print(test_y[:10], 'real number')