# -*- coding: utf-8 -*-
#file location the same as spider
# author yyf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt

# # 预测数据，predict之后的预测结果集
# guess = [1, 0, 1, 2, 1, 0, 1, 0, 1, 0]
# # 真实结果集
# fact = [0, 1, 0, 1, 2, 1, 0, 1, 0, 1]
# # 类别
# classes = list(set(fact))
# # 排序，准确对上分类结果
# classes.sort()
# # 对比，得到混淆矩阵
# confusion = confusion_matrix(guess, fact)
# # 热度图，后面是指定的颜色块，gray也可以，gray_x反色也可以
# plt.imshow(confusion, cmap=plt.cm.Blues)
# # 这个东西就要注意了
# # ticks 这个是坐标轴上的坐标点
# # label 这个是坐标轴的注释说明
# indices = range(len(confusion))
# # 坐标位置放入
# # 第一个是迭代对象，表示坐标的顺序
# # 第二个是坐标显示的数值的数组，第一个表示的其实就是坐标显示数字数组的index，但是记住必须是迭代对象
# plt.xticks(indices, classes)
# plt.yticks(indices, classes)
# # 热度显示仪？就是旁边的那个验孕棒啦
# plt.colorbar()
# # 就是坐标轴含义说明了
# plt.xlabel('guess')
# plt.ylabel('fact')
# # 显示数据，直观些
# for first_index in range(len(confusion)):
#     for second_index in range(len(confusion[first_index])):
#         plt.text(first_index, second_index, confusion[first_index][second_index])
#
# # 显示
# plt.show()
#
#
# # PS:注意坐标轴上的显示，就是classes
# # 如果数据正确的，对应关系显示错了就功亏一篑了
# # 一个错误发生，想要说服别人就更难了
# y_test = [0,0,0,2,1,3,4,1,3,2,1,2,1,4,3,4,0,1,2,2,1,1,1,1,3,2,0,2,3,4,0,1]
# y_pred = [0,2,0,1,1,3,2,1,1,2,1,1,1,2,3,4,2,1,2,0,1,1,0,1,3,2,0,2,0,4,0,1]
# labels = [0,1,2,3,4]
# # 得到混淆矩阵(confusion matrix,简称cm)
# # confusion_matrix 需要的参数：y_true(真实标签),y_pred(预测标签)
# cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
#
# # 打印混淆矩阵
# print("Confusion Matrix: ")
# print(cm)
#
# # 画出混淆矩阵
# # ConfusionMatrixDisplay 需要的参数: confusion_matrix(混淆矩阵), display_labels(标签名称列表)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
# disp.plot()
# plt.show()
#
# # 得到混淆矩阵(confusion matrix,简称cm)
# # confusion_matrix 需要的参数：y_true(真实标签),y_pred(预测标签),normalize(归一化,'true', 'pred', 'all')
# cm = confusion_matrix(y_true=y_test, y_pred=y_pred, normalize='true')
#
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
# disp.plot()
# plt.show()


def draw_cm(y_test,y_pred,labels):
    cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
    # 打印混淆矩阵
    print("Confusion Matrix: ")
    print(cm)

    # 画出混淆矩阵
    # ConfusionMatrixDisplay 需要的参数: confusion_matrix(混淆矩阵), display_labels(标签名称列表)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.show()

    # 得到混淆矩阵(confusion matrix,简称cm)
    # confusion_matrix 需要的参数：y_true(真实标签),y_pred(预测标签),normalize(归一化,'true', 'pred', 'all')
    cm = confusion_matrix(y_true=y_test, y_pred=y_pred, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.show()


def draw_confusion_matrix(pred, real):

    classes = list(set(real))
    classes.sort()
    confusion = confusion_matrix(pred, real)
    plt.imshow(confusion, cmap=plt.cm.Blues)
    indices = range(len(confusion))
    plt.xticks(indices, classes)
    plt.yticks(indices, classes)
    plt.colorbar()
    plt.xlabel('pred')
    plt.ylabel('real')
    for first_index in range(len(confusion)):
        for second_index in range(len(confusion[first_index])):
            plt.text(first_index, second_index, confusion[first_index][second_index])

    plt.show()

