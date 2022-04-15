from torchvision import datasets, transforms
import torch
import numpy as np
from torchvision import transforms
import os
from PIL import Image, ImageOps

class ResizeImage():
    def __init__(self, size):
      if isinstance(size, int):
        self.size = (int(size), int(size))
      else:
        self.size = size
    def __call__(self, img):
      th, tw = self.size
      return img.resize((th, tw))

class PlaceCrop(object):

    def __init__(self, size, start_x, start_y):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.start_x = start_x
        self.start_y = start_y

    def __call__(self, img):
        th, tw = self.size
        return img.crop((self.start_x, self.start_y, self.start_x + tw, self.start_y + th))

def load_training(root_path, dir, batch_size, kwargs):

    transform = transforms.Compose(
        [transforms.Resize([256, 256]),#将尺寸resize为256 x256
         transforms.RandomCrop(224),#随机选取尺寸为224的作为该步骤的输出
         transforms.RandomHorizontalFlip(),#以给定的概率随机水平旋转给定的PIL的图像，默认为0.5
         transforms.ToTensor()])#将img转化为tensor
    data = datasets.ImageFolder(root=root_path + dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)

    # 打印出data的格式#
    # print("打印训练数据的结构\n", data)
    # print(datasets)
    return train_loader

def load_testing(root_path, dir, batch_size, kwargs):
    start_center = (256 - 224 - 1) / 2
    transform = transforms.Compose(
        [transforms.Resize([224, 224]),
         PlaceCrop(224, start_center, start_center),#调用前面定义好的类
         transforms.ToTensor()])
    data = datasets.ImageFolder(root=root_path + dir, transform=transform)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)

    # #打印出data的格式#
    # print("打印测试数据的结构\n", data)
    # print(datasets)
    return test_loader


