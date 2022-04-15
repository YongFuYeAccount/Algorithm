# -*- coding: utf-8 -*-
#file location the same as spider
# author yyf

""""
用的是训练集的30%
辨识错误数量为: 4954
辨识率为: 92.354820 ％
"""""
import csv
from torch.utils.data import DataLoader
import numpy as np
from operator import itemgetter
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn import svm


torch.manual_seed(1)    # reproducible

# Hyper Parameters   #10代37%
EPOCH = 100              #0 train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 32        #批次训练的数据
LR = 0.001  #学习率从0.001

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
trainX1=R_xcsv(r'C:\Users\86182\Desktop\Dataset\dataset2\trainDigits\train0.csv')
trainX2=R_xcsv(r'C:\Users\86182\Desktop\Dataset\dataset2\trainDigits\train1.csv')
trainX3=R_xcsv(r'C:\Users\86182\Desktop\Dataset\dataset2\trainDigits\train2.csv')
trainX4=R_xcsv(r'C:\Users\86182\Desktop\Dataset\dataset2\trainDigits\train3.csv')
trainX5=R_xcsv(r'C:\Users\86182\Desktop\Dataset\dataset2\trainDigits\train4.csv')
trainingMat=np.concatenate((trainX1,trainX2,trainX3,trainX4,trainX5),axis=0)#垂直组合numpy
# trainingMat=np.concatenate((trainX1,trainX2,trainX3),axis=0)#垂直组合numpy

hwLabels1= [0] * len(trainX1)
hwLabels2= [1] * len(trainX1)
hwLabels3= [2] * len(trainX1)
hwLabels4= [3] * len(trainX1)
hwLabels5= [4] * len(trainX1)
train_hwLabels=np.concatenate((hwLabels1,hwLabels2,hwLabels3,hwLabels4,hwLabels5),axis=0)#垂直组合numpy
# train_hwLabels = train_hwLabels.reshape(len(train_hwLabels),) # 修改成一维
# train_hwLabels=np.concatenate((hwLabels1,hwLabels2,hwLabels3),axis=0)#垂直组合numpy


'''
trainingMat=np.array(trainingMat-18) 将训练数据全部减去18
'''
# 打乱索引
#得到的原始数据类型为列表，需要将列表转化为ndarray，在打乱
trainingMat=np.array(trainingMat-18)
train_hwLabels=np.array(train_hwLabels).reshape(len(train_hwLabels),) # 将标签修改成一维


index = [i for i in range(len(trainingMat))] # test_data为测试数据
np.random.shuffle(index) # 打乱索引
trainingMat = trainingMat[index]
train_hwLabels = train_hwLabels[index]


# #将读取的标签和向量数据存到列表中，并reshape他的维度为 总量  x 8 x 8
totaldata =np.array(trainingMat)
# totaldata = totaldata.reshape((len(totaldata),8,8))#knn的输入需要是一维的
totallable = train_hwLabels

# 划分训练集和测试集数据
trainX = totaldata[0:151200]
train_Labels = totallable[0:151200]

testX = totaldata[151201:]
test_Labels = totallable[151201:]

#将nparray 转化为tensor，将其打乱，并重写训练数据集格式为pytorch需要输入的形式
train_x = torch.tensor(trainX).type(torch.FloatTensor)
train_y = torch.tensor(train_Labels, dtype=torch.long)#csv读取int转化为long


 #将标签和输入数据用自定义函数封装
input_traindata = GetLoader(train_x , train_y)
train_data= DataLoader(input_traindata, batch_size=BATCH_SIZE, shuffle=True)


# #划分测试集总的数据为216000

val_x = torch.tensor(testX).type(torch.FloatTensor)
val_y = torch.tensor(test_Labels, dtype=torch.long)#csv读取int转化为long

#将标签和输入数据用自定义函数封装
input_testdata = GetLoader(val_x , val_y)
val_data= DataLoader(input_testdata, batch_size=BATCH_SIZE, shuffle=True)

val_x = val_x.numpy()
val_y= list(val_y.numpy())
train_x = train_x.numpy()
train_y = list(train_y.numpy())
for i in range(len(train_y)):
    int(train_y[i])

for i in range(len(val_y)):
    int(val_y[i])

print(len(train_y),len(train_x))
print(len(val_y),len(val_x))


modelpath = r"C:\Users\86182\Desktop\LearnText\pytorch\test\practice\SVM\model\svm_csv.pkl"
# model=svm.SVC()
# model.fit(train_x,train_y)#模型训练
# torch.save(model,modelpath)
errorCount = 0

model = torch.load(modelpath)
for i in range(len(val_x)):

    classifierResult = model.predict(val_x[i].reshape(1,-1))
    print("SVM得到的辨识结果是: %d, 实际值是: %d" % (classifierResult, val_y[i]))
    if (classifierResult != val_y[i]): errorCount += 1.0
print("\n辨识错误数量为: %d" % errorCount)
print("\n辨识率为: %f ％" % ((1-errorCount/float(len(val_y)))*100))
