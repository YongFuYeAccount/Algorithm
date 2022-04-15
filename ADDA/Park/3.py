# -*- coding: utf-8 -*-
#file location the same as spider
# author yyf


#################保存训练用的数据###############
c = []
d = []
for step, (images, labels) in enumerate(data_loader):
    a = images
    b = labels
    a = a.tolist()
    b = b.float()
    c.append(a)
    d.append(b)

for i in range(len(c)):
    f = open(r"C:\Users\86182\Desktop\Spring\pytorch-adda-master\data\experiment\usps\test_data.csv", 'a', newline='')
    writer = csv.writer(f)
    writer.writerow(c[i])
    f.close()
for i in range(len(d)):
    f = open(r"C:\Users\86182\Desktop\Spring\pytorch-adda-master\data\experiment\usps\test_label.csv", 'a', newline='')
    writer = csv.writer(f)
    writer.writerow(d[i])
    f.close()

