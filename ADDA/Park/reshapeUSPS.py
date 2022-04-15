# -*- coding: utf-8 -*-
#file location the same as spider
# author yyf
# from PIL import Image
# from pylab import *
# #读取图片并转为数组
# im = array(Image. open ( "0001.bmp" ))
# #输出数组的各维度长度以及类型
# print (im.shape,im.dtype)
# print(im)
# #输出位于坐标100,100，颜色通道为r的像素值
# print (im[ 100 , 100 ])


import numpy as np
import csv
datas = []

with open("train_data.csv",encoding="utf-8") as f:
    reader = csv.reader(f)
    header_row = next(reader)
    for row in reader:
        # print(row)
        # print(len(row))
#########将数据reshape为28x28##############
        c = np.array(row)
        c = c.reshape(16, 16)
        e = np.pad(c, (6, 6), 'constant', constant_values=(0, 0))
        e = e.flatten()
        f = e.tolist()
###############对数据进行 mean 和  stand =0.5处理######################
        f = [float(x) for x in f]  # 将字符串转化为浮点数
        for i in range(len(f)):
            f[i] = (f[i] - 0.5) / 0.5  # 将数据进行处理，归一化到-1 --- 1
        f = [str(x) for x in f]  # conver float to str
###########储存所有数据在这个中间变量###################################

        datas.append(f)

print("结束打印")

###########保存处理过后的数据在csv文件中################################
for i in range(len(datas)):
    f = open(r"C:\Users\86182\Desktop\Spring\pytorch-adda-master\data\28x28usps\train_data.csv", 'a',newline='')
    writer = csv.writer(f)
    writer.writerow(datas[i])
    f.close()

