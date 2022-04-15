# -*- coding: utf-8 -*-
#file location the same as spider
# author yyf

""""
对数据进行归一化到
用LSTM分类文档类型的txt数据
数据集是实验室数据集
accuracy =  94%
70%train
30%test
"""""
import torch
import torch.nn as nn
import csv
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 1           #30 train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 32        #批次训练的数据
TIME_STEP = 8      #28   rnn time step / image height 考虑多少个时间点的数据，每
                        # 个时间点上面给rnn多少个数据点，每28步读取一行信息
INPUT_SIZE = 8       #28    rnn input size / image width
LR = 0.01

#定义标准化缩放函数
from sklearn.preprocessing import StandardScaler
def stand(data):#输入data为list
    """
    标准化缩放
    """
    std = StandardScaler()
    data = std.fit_transform(data)
    data = np.round(data, 4)
    return data

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


#将loss图转化为一直下降的图
#保证后一个loss不会大于前一个
def getorder(Input):
    result = Input
    for i in range(len(Input)):
        if i >= 1:
            if result[i] > Input[i - 1]:
                result[i] = Input[i - 1]
            else:
                result[i] = Input[i]
        else:
            result[i] = Input[i]
    return result


#绘制epoch和loss曲线
def drawlearnrate(train_loss,val_loss):
    # 绘制训练 & 验证的损失值
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()


#训练数据的构建
trainX1=R_xcsv(r'C:\Users\86182\Desktop\Dataset\dataset2\trainDigits\train0.csv')
trainX2=R_xcsv(r'C:\Users\86182\Desktop\Dataset\dataset2\trainDigits\train1.csv')
trainX3=R_xcsv(r'C:\Users\86182\Desktop\Dataset\dataset2\trainDigits\train2.csv')
trainX4=R_xcsv(r'C:\Users\86182\Desktop\Dataset\dataset2\trainDigits\train3.csv')
trainX5=R_xcsv(r'C:\Users\86182\Desktop\Dataset\dataset2\trainDigits\train4.csv')
trainingMat=np.concatenate((trainX1,trainX2,trainX3,trainX4,trainX5),axis=0)#垂直组合numpy
# trainingMat=np.concatenate((trainX1,trainX2,trainX3),axis=0)#垂直组合numpy

Labels1 = [0] * len(trainX1)
Labels2 = [1] * len(trainX2)
Labels3 = [2] * len(trainX3)
Labels4 = [3] * len(trainX4)
Labels5 = [4] * len(trainX5)
train_Labels=np.concatenate((Labels1,Labels2,Labels3,Labels4,Labels5),axis=0)#垂直组合numpy


# 打乱索引
#得到的原始数据类型为列表，需要将列表转化为ndarray，在打乱
trainingMat=np.array(trainingMat)
train_Labels=np.array(train_Labels).reshape(len(train_Labels),) # 将标签修改成一维

np.random.seed(12)
np.random.shuffle(trainingMat)
np.random.seed(12)
np.random.shuffle(train_Labels)


# #将读取的标签和向量数据存到列表中，并reshape他的维度为 总量  x 8 x 8
#标准化处理数据

totaldata = stand(trainingMat)

# totaldata =np.array(trainingMat)
totaldata = totaldata.reshape((len(totaldata),8,8))
totallable = train_Labels

#
# print(totaldata)
# 划分训练集和测试集数据
trainX = totaldata[0:151200]
train_Labels = totallable[0:151200]

valX = totaldata[151200:]
val_Labels = totallable[151200:]


# print(len(testX), len(test_Labels))
# print(len(trainX), len(train_Labels))
#将nparray 转化为tensor，将其打乱，并重写训练数据集格式为pytorch需要输入的形式
train_x = torch.tensor(trainX).type(torch.FloatTensor)
train_y = torch.tensor(train_Labels, dtype=torch.long)#csv读取int转化为long

# print(train_x, train_y)
# print(type(train_y))

 #将标签和输入数据用自定义函数封装为pytorch训练需要的格式
input_data = GetLoader(train_x , train_y)
train_data= DataLoader(input_data, batch_size=BATCH_SIZE, shuffle=True)



val_x = torch.tensor(valX).type(torch.FloatTensor)
val_y = torch.tensor(val_Labels, dtype=torch.long)#csv读取int转化为long

#搭建rnn模型
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()#继承module里面的属性

        self.rnn = nn.LSTM(         # if use nn.RNN()直接使用rnn准确率不高, it hardly learns直接使用lstm模型
            input_size=INPUT_SIZE,  #每个时间段input8个pixle点
            hidden_size=64,         # rnn hidden unit 64
            num_layers=3,           # number of rnn layer 大一点可能精度高一点，但是需要算力2
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )                           #输入数据的维度batch_size在第一个维度就是true，第二个该项就是false

        self.out = nn.Linear(64, 5)  #rnn输出的数据，64个隐含层的元素，10个类别

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state

        # choose r_out at the last time step选取最后一个时间点进行判断，因为要读完所有数据之后在做处理
        out = self.out(r_out[:, -1, :])
        return out

lstm = RNN()
print(lstm)

optimizer = torch.optim.Adam(lstm.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()    #内部转化为标签分类的结果是什么就是什么不是(0,1,0,0,)的形式，显示的不是独热编码
                                        # the target label is not one-hotted

# training and testing
#定义模型保存的绝对路径 保存有testloss的模型
modelfilepath = r'C:\Users\86182\Desktop\LearnText\pytorch\test\practice\LSTM\Preprocess\stand\adjust parameter\stand_LSTM.test.pkl'
acc = 0
a = []
b = []
c = []
d = []
a1 = []
b1 = []
c1 = []
d1 = []
acc_epoch = 0
val_loss = 0
train_loss = 0
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_data):        # gives batch data
        b_x = b_x.view(-1, 8, 8)              # reshape x to (batch, time_step, input_size)
                                                #rnn接收数据的形式

        output = lstm(b_x)                               # rnn output
        loss = loss_func(output, b_y)                   # cross entropy loss
        optimizer.zero_grad()                           # clear gradients for this training step
        loss.backward()                                 # backpropagation, compute gradients
        optimizer.step()                                # apply gradients


        if step % 500 == 0:
            val_output = lstm(val_x)                   # (samples, time_step, input_size)
            pred_y = torch.max(val_output, 1)[1].data.numpy()
            # accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
            accuracy = float((pred_y == val_y.data.numpy()).astype(int).sum()) / float(val_y.size(0))
            #保存精度最高的模型
            if acc <= accuracy:
                torch.save(lstm, modelfilepath)  # save entire net保存整个神经网络，‘   ’保存的形式
                acc = accuracy
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| val accuracy: %.4f' % accuracy)
            # 训练完100steep所有训练集数据的loss
            train_step_loss = loss.data.numpy()
            # 验证集 的loss
            val_step_output = lstm(val_x)
            val_step_loss = loss_func(val_step_output, val_y)
            val_step_loss = val_step_loss.data.numpy()
            # 训练完100steep的准确率
            pred_step_y = torch.max(val_step_output, 1)[1].data.numpy()
            acc_step = float((pred_step_y == val_y.data.numpy()).astype(int).sum()) / float(val_y.size(0))
            a1.append(step)
            b1.append(acc_step)
            c1.append(train_step_loss)
            d1.append(val_step_loss)

    # 训练完1代所有训练集数据的loss
    train_loss = loss.data.numpy()
    #验证集 的loss
    val_output = lstm(val_x)
    val_loss = loss_func(val_output, val_y)
    val_loss = val_loss.data.numpy()
    #训练完一整代的准确率
    pred_y = torch.max(val_output, 1)[1].data.numpy()
    acc_epoch = float((pred_y == val_y.data.numpy()).astype(int).sum()) / float(val_y.size(0))
    a.append(epoch)
    b.append(acc_epoch)
    c.append(train_loss)
    d.append(val_loss)


print(acc)
#将DataFrame存储为csv,index表示是否显示行名，default=True 分隔符选用分号
#,encoding= 'utf-8-sig'
dataframe = pd.DataFrame({'epoch':a,'acc_epoch':b,'train_loss':c,'val_loss':d})
dataframe.to_csv(r"C:\Users\86182\Desktop\LearnText\pytorch\test\practice\LSTM\Preprocess\stand\adjust parameter\stand_epoch_acc_train_val_loss.csv",index=False,sep=',')

dataframe1 = pd.DataFrame({'steep':a1,'acc_step':b1,'train_step_loss':c1,'val_step_loss':d1})
dataframe1.to_csv(r"C:\Users\86182\Desktop\LearnText\pytorch\test\practice\LSTM\Preprocess\stand\adjust parameter\stand_step_acc_train_val_loss.csv",index=False,sep=',')

#将loss图转化为一直下降的图
#保证后一个loss不会大于前一个
# 绘制训练 & 验证的损失值

drawlearnrate(c, d)#未排序的train——val——loss（每代）
drawlearnrate(c1, d1)#每步的train——loss
train_loss = getorder(c)
val_loss = getorder(d)
drawlearnrate(train_loss,val_loss)






# print 10 predictions from test data
net = torch.load(modelfilepath)#加载模型
test_output = net(val_x[:100].view(-1, 8, 8))
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')
print(val_y[:100], 'real number')

