# -*- coding: utf-8 -*-
#file location the same as spider
# author yyf
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




#将标签和输入数据用自定义函数封装
input_data = GetLoader(train_x , train_y)
train_data= DataLoader(input_data, batch_size=BATCH_SIZE, shuffle=True)

class AutoEncoder(nn.Module):#一个初始化的过程
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(32*32, 128),#图片信息是32 x 32 像素，所以input，128个隐含层
            nn.Tanh(),#激活函数
            nn.Linear(128, 64),#压缩为64个隐含层
            nn.Tanh(),
            nn.Linear(64, 12),
            nn.Tanh(),
            nn.Linear(12, 3),   # compress to 3 features which can be visualized in plt三个特征所以画出三维图
        )
        self.decoder = nn.Sequential(#encode与decode过程相反
            nn.Linear(3, 12),
            nn.Tanh(),
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 32*32),
            nn.Sigmoid(),       # compress to a range (0, 1)因为训练集的输出是0-1，所以用sigmoid函数将输出压缩到与训练集一致的范围
        )

    def forward(self, x):#将各级连接起来的过程
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


autoencoder = AutoEncoder()

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
loss_func = nn.MSELoss()

for epoch in range(EPOCH):
    for step, (x, b_label) in enumerate(train_data):
        b_x = x.view(-1, 32*32)   # batch x, shape (batch, 28*28)
        b_y = x.view(-1, 32*32)   # batch y, shape (batch, 28*28)

        encoded, decoded = autoencoder(b_x)

        loss = loss_func(decoded, b_y)      # mean square error
        optimizer.zero_grad()               # clear gradients for this training step
        loss.backward()                     # backpropagation, compute gradients
        optimizer.step()                    # apply gradients

        if step % 10 == 0:
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())