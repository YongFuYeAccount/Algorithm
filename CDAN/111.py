import matplotlib.pyplot as plt
import numpy as np

#数据
# v1 = [5, 20, 36, 10, 75, 90]
# v2 = [10, 25, 8, 60, 20, 80]
# attr = ["衬衫", "羊毛衫", "雪纺衫", "裤子", "高跟鞋", "袜子"]

# 设置显示中文
plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
plt.rcParams['axes.unicode_minus']=False     # 正常显示负号

#设置画布大小像素点
plt.figure(figsize=(14,14),dpi=100)

# 绘制直方图
plt.subplot(2,2,1)
data = np.random.randn(1000)# 随机生成（1000）服从正态分布的数据
plt.hist(data,bins=40,facecolor='blue',edgecolor='red')
plt.ylabel("频率/区间")
plt.xlabel("区间")
plt.title("直方图")

# 绘制条形图
plt.subplot(2,2,2)
plt.bar(attr,v1,width=0.4, alpha=0.8, color='red', label="v1")
plt.legend()
plt.ylabel("销量")
plt.xlabel("种类")
plt.title("条形图")

# 绘制条形图
plt.subplot(2,2,3)
plt.bar(attr,v2,width=0.4, alpha=0.8, color='black', label="v2")
plt.legend()
plt.ylabel("销量")
plt.xlabel("种类")
plt.title("条形图")

# 绘制条形图
plt.subplot(2,2,4)
sum =0
for data in v2:
    sum+=data
d =[]
for data in v2:
    d.append(data/sum)
explode = [0.06,0,0,0,0.05,0]
plt.pie(x=d,explode=explode,labels=attr,autopct = '%3.2f%%', colors=('b', 'g', 'r', 'c', 'm', 'y'))
plt.legend()
plt.title("饼图")

# plt.savefig("D:\\StudyDemo\\IDEA\\PythonDemo\\MatplotlibPaint\\SaveData\\subplot.png")
plt.show()
