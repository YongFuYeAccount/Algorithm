# -*- coding: utf-8 -*-
#file location the same as spider
# author yyf
""""
固定超参数， 创建pytorch训练数据集，以及测试数据集，
搭建网络模型，训练测试
用的数据集是r'C:\Users\86182\Desktop\GraduateDesign\Aug\Infrared image\trainingDigits
里面的人体红外数据集
"""""



import torch
import torch.nn as nn
import torchvision   #torchvision里面有一部分是数据集
import matplotlib.pyplot as plt
from os import listdir
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image


# Hyper Parameters
EPOCH = 5          # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 8
LR = 0.001              # learning rate

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



def img2vector(filename):
    img = Image.open(filename)
    arr=np.asarray(img)
    return arr

hwLabels = []
trainingFileList = listdir(r'C:\Users\86182\Desktop\GraduateDesign\Aug\Infrared image\trainingDigits')
m = len(trainingFileList)
trainingMat = [];
for i in range(m):
    fileNameStr = trainingFileList[i]
    fileStr = fileNameStr.split('.')[0]
    classNumStr = int(fileStr[0])
    hwLabels.append(classNumStr)
    trainingMat.append(img2vector(r'C:\Users\86182\Desktop\GraduateDesign\Aug\Infrared image\trainingDigits/%s' % fileNameStr))



trainX=np.array(trainingMat)
trainX=trainX.reshape((trainX.shape[0],1,trainX.shape[1],trainX.shape[2]))

#将nparray 转化为tensor，将其打乱，并重写训练数据集格式为pytorch需要输入的形式
train_x = torch.tensor(trainX).type(torch.FloatTensor)
train_y = torch.tensor(hwLabels)
#将标签和输入数据用自定义函数封装
input_data = GetLoader(train_x , train_y)
train_data= DataLoader(input_data, batch_size=BATCH_SIZE, shuffle=True)


##测试集的数据处理
test_hwLabels = []
testFileList = listdir(r'C:\Users\86182\Desktop\GraduateDesign\Aug\Infrared image\testDigits')
m = len(testFileList)
testMat = [];
for i in range(m):
    fileNameStr = testFileList[i]
    fileStr = fileNameStr.split('.')[0]
    classNumStr = int(fileStr[0])
    test_hwLabels.append(classNumStr)
    # hwLabels.append(to_categorical(classNumStr, 10))  # 独热编码利用二进制位的方式标为10类从一维变成十维
    testMat.append(img2vector(r'C:\Users\86182\Desktop\GraduateDesign\Aug\Infrared image\testDigits/%s' % fileNameStr))

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
testX=testX.reshape((testX.shape[0],1,testX.shape[1],testX.shape[2]))  #

#将输入数据和标签转化为pytorch要求的tensor格式
test_x = torch.tensor(testX).type(torch.FloatTensor)
test_y = torch.tensor(test_hwLabels)

#建立cnn网络模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 150, 150)维度 图片的宽和高
            nn.Conv2d(
                in_channels=1,              # input 通道数
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 150, 150)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=3),    # choose max value in 3x3 area, output shape (16, 50, 50)
        )
        self.conv2 = nn.Sequential(         # input shape (16, 50, 50)
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32,50, 50)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 25, 25)
        )
        self.conv3 = nn.Sequential(  # input shape (32, 25, 25)
            nn.Conv2d(32, 32, 5, 1, 2),  # output shape (32,25, 25)
            nn.ReLU(),  # activation
            nn.MaxPool2d(1),  # output shape (32, 25,25)maxpool将25维度转化为5
        )
        self.out = nn.Linear(32 * 25 * 25, 6)   # fully connected layer, output 6 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)                  #(batch_size, 32 * 7 * 7)
        x = self.conv3(x)
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
            test_output, last_layer = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            # print(pred_y,type(pred_y),len(pred_y))#预测的标签有1934个出现的原因主要是因为数据出错了
            # print(test_y.data.numpy(),type(test_y.data.numpy()),len(test_y.data.numpy()))#实际的标签有946个
            # print(float(test_y.size(0)))
            # print(np.sum(pred_y==test_y.data.numpy()))
            # print(float((pred_y == test_y.data.numpy()).astype(int).sum()))
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
        #     # # # # if HAS_SK:
            # #     # Visualization of trained flatten layer (T-SNE)
            #     tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
            #     plot_only = 500
            #     low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
            #     labels = test_y.numpy()[:plot_only]
            #     plot_with_labels(low_dim_embs, labels)
# plt.ioff()
# print 10 predictions from test data
test_output, _ = cnn(test_x[:50])
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')
print(test_y[:50].numpy(), 'real number')
