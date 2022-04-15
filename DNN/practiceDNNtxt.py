# -*- coding: utf-8 -*-
#file location the same as spider
# author yyf

"""""
输入数据是32*32，训练数据和测试数据都添加到了，
pytorch构建的数据集中标签和数据都放入一起
输入数据为floattensor ，标签为longtensor
Accuracy = 97%
"""""


import torch
import torch.nn as nn
from os import listdir
import numpy as np
from torch.utils.data import DataLoader



# Hyper Parameters
EPOCH = 10#1           # train the training data n times, to save time, we just train 1 epoch
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
# trainX=trainX.reshape((trainX.shape[0],1,trainX.shape[1],trainX.shape[2]))

#将nparray 转化为tensor，将其打乱，并重写训练数据集格式为pytorch需要输入的形式
train_x = torch.tensor(trainX).type(torch.FloatTensor)
train_y = torch.tensor(hwLabels)

# print(train_x, train_y)


#将标签和输入数据用自定义函数封装
input_data = GetLoader(train_x , train_y)
train_data= DataLoader(input_data, batch_size=BATCH_SIZE, shuffle=True)
# print(input_data,train_data)
# print(trainX)


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
# testX=testX.reshape((testX.shape[0],1,testX.shape[1],testX.shape[2]))

#将输入数据和标签转化为pytorch要求的tensor格式
test_x = torch.tensor(testX).type(torch.FloatTensor)
test_y = torch.tensor(test_hwLabels).type(torch.LongTensor)

#将test_data进行封装
input_data2 = GetLoader(test_x , test_y)
test_data= DataLoader(input_data2, batch_size=BATCH_SIZE, shuffle=True)

#搭建模型
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(1024, 200),#输入是32 x 32 =1024,输入的特征是1024，隐含层200 ；in_features=1024, out_features=200, bias=True)
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 200),#in_features=200, out_features=200
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 10),#输出十个特征 ；in_features=200, out_features=10, bias=True)
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

# training and testing
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_data):   # gives batch data, normalize x when iterate train_loader
                                                   #b_x是上面构造数据集里面的的数据，为batchsize X 32 X 32
        b_x = b_x.view(-1, 32 * 32)  # reshape x to (batch, input_feature)
                                    #b_x一个批次里面的所有输入数据组成的张量 ,b_y该batchsize里面的数据的标签张量 batchsize X (32*32=1024)
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
        data_x = data_x.view(-1, 32 * 32)  # batch x 输入数据28*28
        pred_y= dnn(data_x)
        test_loss += loss_func(pred_y, data_y).item()

        pred = pred_y.data.max(1)[1]
        correct += pred.eq(data_y.data).sum()

    test_loss /= len(test_data.dataset)
    print('\nTest set: Average loss: {:.4f},Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_data.dataset), 100. * correct / len(test_data.dataset)))

#         if step % 10 == 0:
#             test_output = dnn(test_x)                   # (samples, time_step, input_size)
#             pred_y = torch.max(test_output, 1)[1].data.numpy()
#             # accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
#             accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
#             print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
#
# # print 10 predictions from test data
# test_output = rnn(test_x[:10].view(-1, 32*32))
# pred_y = torch.max(test_output, 1)[1].data.numpy()
# print(pred_y, 'prediction number')
# print(test_y[:10], 'real number')
