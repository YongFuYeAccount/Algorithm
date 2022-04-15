# -*- coding: utf-8 -*-
#file location the same as spider
# author yyf
# 两种方法都能打开
import pickle
import numpy as np

f = open('testing_s_statistic.pkl','rb')
data = pickle.load(f)
print(data)

# img_path = './train_data.pkl'
# img_data = np.load(img_path)
# print(img_data)

