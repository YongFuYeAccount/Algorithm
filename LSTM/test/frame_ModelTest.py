import torch
import torch.nn as nn
import csv
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize

random_state = np.random.RandomState(0)
torch.manual_seed(1)    # reproducible
# Hyper Parameters
# Hyper Parameters
EPOCH = 20           #30 train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 32        #批次训练的数据
TIME_STEP = 64 * 5      #28   rnn time step / image height 考虑多少个时间点的数据，每
                        # 个时间点上面给rnn多少个数据点，每28步读取一行信息
INPUT_SIZE = 1      #28    rnn input size / image width
LR = 0.01


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


#画混淆矩阵
def draw_cm(y_test,y_pred,labels):
    cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
    # 打印混淆矩阵
    print("Confusion Matrix: ")
    print(cm)

    # 画出混淆矩阵
    # ConfusionMatrixDisplay 需要的参数: confusion_matrix(混淆矩阵), display_labels(标签名称列表)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

    # 得到混淆矩阵(confusion matrix,简称cm)
    # confusion_matrix 需要的参数：y_true(真实标签),y_pred(预测标签),normalize(归一化,'true', 'pred', 'all')
    cm = confusion_matrix(y_true=y_test, y_pred=y_pred, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

#画混淆矩阵
# 相关库
def plot_matrix(y_true, y_pred, labels_name, title=None, thresh=0.8, axis_labels=None):
# 利用sklearn中的函数生成混淆矩阵并归一化
    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels_name, sample_weight=None)  # 生成混淆矩阵
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化

# 画图，如果希望改变颜色风格，可以改变此部分的cmap=pl.get_cmap('Blues')处
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.colorbar()  # 绘制图例

# 图像标题
    if title is not None:
        pl.title(title)
# 绘制坐标
    num_local = np.array(range(len(labels_name)))
    if axis_labels is None:
        axis_labels = labels_name
    plt.xticks(num_local, axis_labels, rotation=45)  # 将标签印在x轴坐标上， 并倾斜45度
    plt.yticks(num_local, axis_labels)  # 将标签印在y轴坐标上
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# 将百分比打印在相应的格子内，大于thresh的用白字，小于的用黑字
    for i in range(np.shape(cm)[0]):
        for j in range(np.shape(cm)[1]):
            if int(cm[i][j] * 100 + 0.5) > 0:
                plt.text(j, i, format(int(cm[i][j] * 100 + 0.5), 'd') + '%',
                        ha="center", va="center",
                        color="white" if cm[i][j] > thresh else "black")  # 如果要更改颜色风格，需要同时更改此行
# 显示
    plt.show()


#保存训练的概率为csv
def keepresult(savefilepath, headerlines, real_Labels, pred_class, totalprobality):
    # 保存csv的地址，csv的列名，实际的标签（list），预测的标签（list），各个列别的标签(list)
    csvFile = open(savefilepath, "w",newline="")  # 创建csv文件
    writer = csv.writer(csvFile)  # 创建写的对象
    # 先写入columns_name
    writer.writerow(headerlines)  # 写入列的名称
    for i in range(len(list(real_Labels))):
        probability = list(totalprobality[i])  # 将概率np转化为list
        writer.writerow(
            [real_Labels[i], pred_class[i], probability[0], probability[1], probability[2], probability[3],
             probability[4]])
    csvFile.close()
# 绘制精确率-召回率曲线
def plot_precision_recall(recall, precision):
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.plot(recall, precision, linewidth=2)
    plt.xlim([0.0, 1])
    plt.ylim([0.0, 1.05])
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('precision-recall carve')
    plt.show();

#画pr曲线
def drawPre_recall(test_y, y_score):  # test_y为测试数据的类别，y_score是每个类别的概率
    y = test_y
    # 使用label_binarize让数据成为类似多标签的设置
    Y = label_binarize(y, classes=[0, 1, 2, 3, 4])
    n_classes = Y.shape[1]

    # 对每个类别
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(Y[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(Y[:, i], y_score[:, i])

    # 一个"微观平均": 共同量化所有课程的分数
    precision["micro"], recall["micro"], _ = precision_recall_curve(Y.ravel(), y_score.ravel())

    average_precision["micro"] = average_precision_score(Y, y_score, average="micro")

    print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))

    plt.figure()
    plt.step(recall['micro'], precision['micro'], where='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.01])
    plt.title(
        'Average precision score, micro-averaged over all classes: AP={0:0.2f}'.format(average_precision["micro"]))
    plt.show()
#将csv文件保存为txt
def keeptxt(filepath_csv, filepath_txt):
    data = pd.read_csv(filepath_csv, encoding='gbk')
    with open(filepath_txt, 'w', encoding='gbk') as f:
        for line in data.values:
            f.write(
                (str(line[0]) + '\t' + str(line[1]) + '\t' + str(line[2]) + '\t'
                 + str(line[3]) + '\t' + str(line[4])+ '\t' + str(line[5])+ '\t' + str(line[6])))
            f.write("\n")

#训练数据的构建
testX1=R_xcsv(r'C:\Users\86182\Desktop\Dataset\dataset2\testDigits\test0.csv')
testX2=R_xcsv(r'C:\Users\86182\Desktop\Dataset\dataset2\testDigits\test1.csv')
testX3=R_xcsv(r'C:\Users\86182\Desktop\Dataset\dataset2\testDigits\test2.csv')
testX4=R_xcsv(r'C:\Users\86182\Desktop\Dataset\dataset2\testDigits\test3.csv')
testX5=R_xcsv(r'C:\Users\86182\Desktop\Dataset\dataset2\testDigits\test4.csv')
testMat=np.concatenate((testX1,testX2,testX3,testX4,testX5),axis=0)#垂直组合numpy


#将五个数据作为一个样本
totaldata = np.array(testMat - 18)
datalength = int(len(totaldata)/ 5)
totaldata = totaldata.reshape(datalength,1,64 * 5)

number = int(len(testX1) / 5)
Labels1 = [0] * number
Labels2 = [1] * number
Labels3 = [2] * number
Labels4 = [3] * number
Labels5 = [4] * number
test_Labels=np.concatenate((Labels1, Labels2, Labels3, Labels4, Labels5),axis=0)#垂直组合numpy

# 打乱索引
#得到的原始数据类型为列表，需要将列表转化为ndarray，在打乱
totaldata = np.array(totaldata)
totallable = np.array(test_Labels).reshape(len(test_Labels),) # 将标签修改成一维

np.random.seed(12)
np.random.shuffle(totaldata)
np.random.seed(12)
np.random.shuffle(totallable)

#将nparray 转化为tensor，将其打乱，并重写训练数据集格式为pytorch需要输入的形式
test_x = torch.tensor(totaldata).type(torch.FloatTensor)
test_y = torch.tensor(totallable, dtype=torch.long)#csv读取int转化为long

# Labels1= [0] * len(testX1)
# Labels2= [1] * len(testX2)
# Labels3= [2] * len(testX3)
# Labels4= [3] * len(testX4)
# Labels5= [4] * len(testX5)
# test_Labels=np.concatenate((Labels1,Labels2,Labels3,Labels4,Labels5),axis=0)#垂直组合numpy
#
# print(testMat, test_Labels)

# # 打乱索引
# #得到的原始数据类型为列表，需要将列表转化为ndarray，在打乱
# testMat=np.array(testMat)
# test_Labels=np.array(test_Labels) # 将标签修改成一维
#
# index = [i for i in range(len(testMat))] # test_data为测试数据
# np.random.shuffle(index) # 打乱索引
# testMat = testMat[index]
# test_Labels = test_Labels[index]
#
#
# # #将读取的标签和向量数据存到列表中，并reshape他的维度为 总量  x 8 x 8
# totaldata = np.array(testMat-18)
# testX = totaldata.reshape((len(totaldata),8,8))


# #将nparray 转化为tensor，将其打乱，并重写训练数据集格式为pytorch需要输入的形式
# test_x = torch.tensor(testX).type(torch.FloatTensor)
# test_y = torch.tensor(test_Labels, dtype=torch.long)#csv读取int转化为long
# print(test_x, test_y)
#搭建rnn模型
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()#继承module里面的属性

        self.rnn = nn.LSTM(         # if use nn.RNN()直接使用rnn准确率不高, it hardly learns直接使用lstm模型
            input_size=INPUT_SIZE,  #每个时间段input8个pixle点
            hidden_size=64,         # rnn hidden unit
            num_layers=3,           # number of rnn layer 大一点可能精度高一点，但是需要算力
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

# testing
#定义模型保存的绝对路径
modelfilepath = r"C:\Users\86182\Desktop\LearnText\pytorch\test\practice\LSTM\Frames\model\node_32\30_epoch_final_lstm.pkl"
save_csvfilepath = r"C:\Users\86182\Desktop\LearnText\pytorch\test\practice\LSTM\Frames\test\ClassificateOdds.csv"
save_txtfilepath = r"C:\Users\86182\Desktop\LearnText\pytorch\test\practice\LSTM\Frames\test\ClassificateOdds.txt"
net = torch.load(modelfilepath)#加载模型

#测试所有数据
test_output = net(test_x.view(-1,64 , 5 ))
pred_y = torch.max(test_output, 1)[1].data.numpy()

print(pred_y)
print(len(pred_y))
print(test_y.data.numpy())
print(len(test_y.data.numpy()))
#计算测试的总体准确率
accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
print(accuracy)

#得到分类过程的分类概率
probability = torch.nn.functional.softmax(test_output,dim=1)#计算softmax，即该图片属于各类的概率
# max_value,index = torch.max(probability,1)#找到最大概率对应的索引号，该图片即为该索引号对应的类别（输出是tensor）

#保存数据需要的参数
totalprobality = np.round(probability.detach().numpy(), 3)#保留概率的三位小数
test_Labels = list(test_Labels)
headerlines = ["实际", "预测", "empty","sitting","standing","walking","lying"]
real_Labels = test_Labels
pred_class = list(pred_y)

# #保存训练过程的数据
# #将测试得到的分类概率保存到csv文件中
# keepresult(save_csvfilepath, headerlines, real_Labels, pred_class, totalprobality)
# #将保存的csv文件转化为txt文件
# keeptxt(save_csvfilepath, save_txtfilepath)
#
classes = ["empty","sitting","standing","walking","lying"]# classes = list(reversed(classes))print(a,list(reversed(a)))
label = [0,1,2,3,4]
# 计算precision, recall, F1-score, support的两种方法
print(classification_report(list(test_y), list(pred_y), target_names=classes))
precision, recall, fscore, support = precision_recall_fscore_support(list(test_y), list(pred_y))

#画图
#划分预测数据，实际数据，以及标签的种类，画混淆矩阵
#画混淆矩阵  # #将a倒序排列

draw_cm(list(pred_y), list(test_y), classes)#调用画混淆矩阵函数
plot_matrix(list(test_y), list(pred_y), label, title=None, thresh=0.8, axis_labels=None)#画混淆矩阵方法二

# #画pr曲线
drawPre_recall(test_y, totalprobality)

# #打印各个类别的结果
# import collections
# data_count =collections.Counter(pred_y)
# print(data_count)
#
# print(pred_y, 'prediction number')
# print(test_y, 'real number')
#
#
#
#
#
