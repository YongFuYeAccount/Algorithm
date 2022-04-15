# -*- coding: utf-8 -*-
#file location the same as spider
# author yyf
import numpy as np
import csv
datas = []

with open("train_data.csv",encoding="utf-8") as f:
    reader = csv.reader(f)
    header_row = next(reader)
    for row in reader:
        # print(row)
        # print(len(row))
        c = np.array(row)
        c = c.reshape(16, 16)
        e = np.pad(c, (6, 6), 'constant', constant_values=(0, 0))
        e = e.flatten()
        f = e.tolist()

        f = [float(x) for x in f]#将字符串转化为浮点数
        for i in range(len(f)):
            f[i] = (f[i] - 0.5) / 0.5#将数据进行处理，归一化到-1 --- 1
        f = [str(x) for x in f]#conver float to str

        # print(len(f))
        # # print(f)

        datas.append(f)
        # print(f)
        # with open("output.csv", 'a+', encoding='utf8',newline='') as name:
        #     # 写入文件时增加换行符，保证每个元素位于一行
        #     name.write(e + '\t')
print("结束打印")