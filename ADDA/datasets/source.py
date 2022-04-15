# -*- coding: utf-8 -*-
#file location the same as spider
# author yyf
import csv
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

# torch.manual_seed(1)    # reproducible
# # Hyper Parameters
# EPOCH = 1#30           # train the training data n times, to save time, we just train 1 epoch
# BATCH_SIZE = 32
# LR = 0.01              # learning rate

#
# #读csv函数(读数据)
# def R_xcsv(Filedirectory):
#     csv_file = open(Filedirectory, encoding='utf-8')  # 打开csv文件
#     csv_reader_lines = csv.reader(csv_file)  # 逐行读取csv文件
#     date = []  # 创建列表准备接收csv各行数据
#     renshu = 0
#     for one_line in csv_reader_lines:
#         date.append(one_line)  # 将读取的csv分行数据按行存入列表‘date’中
#         renshu = renshu + 1  # 统计行数
#
#     #读取的date是字符串，将其转化为数值
#     for i in range(len(date)):
#         date[i] = list(map(float, date[i]))
#     # for i in range(len(date)):
#     #     date[i] = np.array(date[i]).reshape(8, 8)#将列表的元素转化为8 x 8
#     # date = np.array(date, dtype=float)  # trainX为待转化的列表
#     return date   #返回的数据是浮点型存在list中
#
#
# #读标签函数
# def R_ycsv(Filedirectory):
#     csv_file = open(Filedirectory, encoding='utf-8')  # 打开csv文件
#     csv_reader_lines = csv.reader(csv_file)  # 逐行读取csv文件
#     date = []  # 创建列表准备接收csv各行数据
#     renshu = 0
#     for one_line in csv_reader_lines:
#         date.append(one_line)  # 将读取的csv分行数据按行存入列表‘date’中
#         renshu = renshu + 1  # 统计行数
#
#     for i in range(len(date)):#将读取的字符串转化为数值
#         date[i] = list(map(int, date[i]))
#     # date = np.array(date, dtype=float)  # trainX为待转化的列表
#     return date
#
# #构建pytorch行驶本数据集函数
# # 定义GetLoader类，继承Dataset方法，并重写__getitem__()和__len__()方法
# class GetLoader(torch.utils.data.Dataset):
# 	# 初始化函数，得到数据
#     def __init__(self, data_root, data_label):
#         self.data = data_root
#         self.label = data_label
#     # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
#     def __getitem__(self, index):
#         data = self.data[index]
#         labels = self.label[index]
#         return data, labels
#     # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
#     def __len__(self):
#         return len(self.data)
#
# #训练数据的构建 读取数据集中的所有数据
# Mat1=R_xcsv(r'C:\Users\86182\Desktop\Dataset\dataset2\trainDigits\d0.csv')
# Mat2=R_xcsv(r'C:\Users\86182\Desktop\Dataset\dataset2\trainDigits\d1.csv')
# Mat3=R_xcsv(r'C:\Users\86182\Desktop\Dataset\dataset2\trainDigits\d2.csv')
# Mat4=R_xcsv(r'C:\Users\86182\Desktop\Dataset\dataset2\trainDigits\d3.csv')
# Mat5=R_xcsv(r'C:\Users\86182\Desktop\Dataset\dataset2\trainDigits\d4.csv')
# totalMat=np.concatenate((Mat1,Mat2,Mat3,Mat4,Mat5),axis=0)#垂直组合numpy
# # trainingMat=np.concatenate((trainX1,trainX2,trainX3),axis=0)#垂直组合numpy
#
# Labels1= [0] * len(Mat1)
# Labels2= [1] * len(Mat2)
# Labels3= [2] * len(Mat3)
# Labels4= [3] * len(Mat4)
# Labels5= [4] * len(Mat5)
# total_Labels=np.concatenate((Labels1,Labels2,Labels3,Labels4,Labels5),axis=0)#垂直组合numpy
#
#
# # 打乱索引
# #得到的原始数据类型为列表，需要将列表转化为ndarray，在打乱
# totalMat=np.array(totalMat)
# total_Labels=np.array(total_Labels).reshape(len(total_Labels),) # 将标签修改成一维
#
#
# index = [i for i in range(len(totalMat))] # test_data为测试数据
# np.random.shuffle(index) # 打乱索引
# totalMat = totalMat[index]
# total_Labels = total_Labels[index]
#
#
# # # #将读取的标签和向量数据存到列表中，并reshape他的维度为 总量 x 1 x 8 x 8
# # trainX=np.array(trainingMat)
# # trainX = trainX.reshape((len(trainX),1,8,8))
# #
# # # print(trainX,train_hwLabels)
# # a = trainX
# # b = train_hwLabels
#
# # #将读取的标签和向量数据存到列表中，并reshape他的维度为 总量 x 1 x 8 x 8
# totaldata =np.array(trainingMat - 18)#对数据进行预处理
# totaldata = totaldata.reshape((len(totaldata),1,8,8))
# totallable = total_Labels
#
# # 划分训练集和测试集数据70%训练集，30%测试集
# trainX = totaldata[0:151200]
# train_Labels = totallable[0:151200]
#
# testX = totaldata[151201:]
# test_Labels = totallable[151201:]
#
# ##封装训练集数据：
# #将nparray 转化为tensor，将其打乱，并重写训练数据集格式为pytorch需要输入的形式
# train_x = torch.tensor(trainX).type(torch.FloatTensor)
# train_y = torch.tensor(train_Labels, dtype=torch.long)#csv读取int转化为long
#
#  #将标签和输入数据用自定义函数封装
# input_train_data = GetLoader(train_x , train_y)
# source_train_data_loader = DataLoader(input_train_data, batch_size=BATCH_SIZE, shuffle=True)
#
#
# #测试集数据
# test_x = torch.tensor(testX).type(torch.FloatTensor)
# test_y = torch.tensor(test_Labels, dtype=torch.long)#csv读取int转化为long
#
# input_test_data = GetLoader(test_x , test_y)
# source_test_data_loader = DataLoader(input_test_data, batch_size=BATCH_SIZE, shuffle=True)

def source_loader(BATCH_SIZE):
    # 读csv函数(读数据)
    def R_xcsv(Filedirectory):
        csv_file = open(Filedirectory, encoding='utf-8')  # 打开csv文件
        csv_reader_lines = csv.reader(csv_file)  # 逐行读取csv文件
        date = []  # 创建列表准备接收csv各行数据
        renshu = 0
        for one_line in csv_reader_lines:
            date.append(one_line)  # 将读取的csv分行数据按行存入列表‘date’中
            renshu = renshu + 1  # 统计行数

        # 读取的date是字符串，将其转化为数值
        for i in range(len(date)):
            date[i] = list(map(float, date[i]))
        # for i in range(len(date)):
        #     date[i] = np.array(date[i]).reshape(8, 8)#将列表的元素转化为8 x 8
        # date = np.array(date, dtype=float)  # trainX为待转化的列表
        return date  # 返回的数据是浮点型存在list中

    # 读标签函数
    def R_ycsv(Filedirectory):
        csv_file = open(Filedirectory, encoding='utf-8')  # 打开csv文件
        csv_reader_lines = csv.reader(csv_file)  # 逐行读取csv文件
        date = []  # 创建列表准备接收csv各行数据
        renshu = 0
        for one_line in csv_reader_lines:
            date.append(one_line)  # 将读取的csv分行数据按行存入列表‘date’中
            renshu = renshu + 1  # 统计行数

        for i in range(len(date)):  # 将读取的字符串转化为数值
            date[i] = list(map(int, date[i]))
        # date = np.array(date, dtype=float)  # trainX为待转化的列表
        return date

    # 构建pytorch行驶本数据集函数
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

    # 训练数据的构建 读取数据集中的所有数据
    Mat1 = R_xcsv(r'C:\Users\86182\Desktop\Dataset\dataset2\trainDigits\d0.csv')
    Mat2 = R_xcsv(r'C:\Users\86182\Desktop\Dataset\dataset2\trainDigits\d1.csv')
    Mat3 = R_xcsv(r'C:\Users\86182\Desktop\Dataset\dataset2\trainDigits\d2.csv')
    Mat4 = R_xcsv(r'C:\Users\86182\Desktop\Dataset\dataset2\trainDigits\d3.csv')
    Mat5 = R_xcsv(r'C:\Users\86182\Desktop\Dataset\dataset2\trainDigits\d4.csv')
    totalMat = np.concatenate((Mat1, Mat2, Mat3, Mat4, Mat5), axis=0)  # 垂直组合numpy
    # trainingMat=np.concatenate((trainX1,trainX2,trainX3),axis=0)#垂直组合numpy

    Labels1 = [0] * len(Mat1)
    Labels2 = [1] * len(Mat2)
    Labels3 = [2] * len(Mat3)
    Labels4 = [3] * len(Mat4)
    Labels5 = [4] * len(Mat5)
    total_Labels = np.concatenate((Labels1, Labels2, Labels3, Labels4, Labels5), axis=0)  # 垂直组合numpy

    # 打乱索引
    # 得到的原始数据类型为列表，需要将列表转化为ndarray，在打乱
    totalMat = np.array(totalMat)
    total_Labels = np.array(total_Labels).reshape(len(total_Labels), )  # 将标签修改成一维

    index = [i for i in range(len(totalMat))]  # test_data为测试数据
    np.random.shuffle(index)  # 打乱索引
    totalMat = totalMat[index]
    total_Labels = total_Labels[index]

    # # #将读取的标签和向量数据存到列表中，并reshape他的维度为 总量 x 1 x 8 x 8
    # trainX=np.array(trainingMat)
    # trainX = trainX.reshape((len(trainX),1,8,8))
    #
    # # print(trainX,train_hwLabels)
    # a = trainX
    # b = train_hwLabels

    # #将读取的标签和向量数据存到列表中，并reshape他的维度为 总量 x 1 x 8 x 8
    totaldata = np.array(trainingMat - 18)  # 对数据进行预处理
    totaldata = totaldata.reshape((len(totaldata), 1, 8, 8))
    totallable = total_Labels

    # 划分训练集和测试集数据70%训练集，30%测试集
    trainX = totaldata[0:151200]
    train_Labels = totallable[0:151200]

    testX = totaldata[151201:]
    test_Labels = totallable[151201:]

    ##封装训练集数据：
    # 将nparray 转化为tensor，将其打乱，并重写训练数据集格式为pytorch需要输入的形式
    train_x = torch.tensor(trainX).type(torch.FloatTensor)
    train_y = torch.tensor(train_Labels, dtype=torch.long)  # csv读取int转化为long

    # 将标签和输入数据用自定义函数封装
    input_train_data = GetLoader(train_x, train_y)
    source_train_data_loader = DataLoader(input_train_data, batch_size=BATCH_SIZE, shuffle=True)

    # 测试集数据
    test_x = torch.tensor(testX).type(torch.FloatTensor)
    test_y = torch.tensor(test_Labels, dtype=torch.long)  # csv读取int转化为long

    input_test_data = GetLoader(test_x, test_y)
    source_test_data_loader = DataLoader(input_test_data, batch_size=BATCH_SIZE, shuffle=True)

    return source_train_data_loader,  source_test_data_loader
