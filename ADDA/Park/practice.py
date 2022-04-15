# -*- coding: utf-8 -*-
#file location the same as spider
# author yyf
import h5py
import pandas as pd
import numpy as np
import pandas as pd

import numpy
# 读取 USPS数据集
def pre_handle():
    with h5py.File(r'C:\Users\86182\Desktop\Spring\pytorch-adda-master\data\USPS\usps.h5') as hf:
            train = hf.get('train')
            x_train = train.get('data')[:]
            y_train = train.get('target')[:]
            test = hf.get('test')
            x_test = test.get('data')[:]
            y_test = test.get('target')[:]

    train_data=pd.DataFrame(x_train)
    train_label=pd.DataFrame(y_train)
    test_data=pd.DataFrame(x_test)
    test_label=pd.DataFrame(y_test)
    return train_data,train_label,test_data,test_label


train_data,train_label,test_data,test_label = pre_handle()

print(train_data)
print(type(train_data))

train_data = np.mat(train_data)
train_label = np.mat(train_label)
test_data = np.mat(test_data)
test_label = np.mat(test_label)
train_label = train_label.astype(int)
test_label = test_label.astype(int)
print(test_label)
print(type(test_label))
train_data = train_data *255
# list(train_data)
# list(train_label)
# print(type(train_data))
# dataframe = pd.DataFrame({'a_name':train_data,'b_name':train_label})#,'c_name':test_data,'d_name':test_label
# dataframe.to_csv(r"C:\Users\86182\Desktop\Spring\pytorch-adda-master\Park\1.csv",index=False,sep=',')
# print("##########")
# my_matrix = np.loadtxt(open(r"C:\Users\86182\Desktop\Spring\pytorch-adda-master\Park\1.csv","rb"),delimiter=",",skiprows=0)

np.savetxt(r"C:\Users\86182\Desktop\Spring\pytorch-adda-master\Park\1.csv", train_data, delimiter = ',')
# np.savetxt(r"C:\Users\86182\Desktop\Spring\pytorch-adda-master\Park\train_label.csv", train_label, delimiter = ',')
# np.savetxt(r"C:\Users\86182\Desktop\Spring\pytorch-adda-master\Park\test_data.csv", test_data, delimiter = ',')
# np.savetxt(r"C:\Users\86182\Desktop\Spring\pytorch-adda-master\Park\test_label.csv", test_label, delimiter = ',')
# print("##################")
# print(train_label)
# print(type(train_label))
# print(len(train_label))
# print("##################")
# print(len(test_data))
# print(type(test_data))
# train_data = np.mat(train_data)
# train_label = np.mat(train_label)
# test_data = np.mat(test_data)
# test_label = np.mat(test_label)
# train_label = train_label.astype(int)
# test_label = test_label.astype(int)