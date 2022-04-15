# -*- coding: utf-8 -*-
#file location the same as spider
# author yyf
import csv
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

torch.manual_seed(1)    # reproducible


def get_train_data(BATCH_SIZE):
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
    train_data = R_xcsv(r'C:\Users\86182\Desktop\Spring\pytorch-adda-master\data\28x28usps\train_data.csv')
    train_Label = R_ycsv(r'C:\Users\86182\Desktop\Spring\pytorch-adda-master\data\28x28usps\train_label.csv')

    # print(train_data)
    # print("##################")
    # print(train_Label)

    train_data = np.array(train_data)
    train_data = train_data.reshape((len(train_data), 1, 28, 28))

    # print(train_data)
    # print(train_Label)

    train_data = torch.tensor(train_data).type(torch.FloatTensor)
    train_label = torch.tensor(train_Label, dtype=torch.long)  # csv读取int转化为long

    # 将标签和输入数据用自定义函数封装
    input_data = GetLoader(train_data, train_label)
    usps_train_data_loader = DataLoader(input_data, batch_size=BATCH_SIZE, shuffle=True)

    return usps_train_data_loader