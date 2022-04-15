import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import pdb

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

def CDAN(input_list, ad_net, entropy=None, coeff=None, random_layer=None):
    # ad_net.cuda()
    softmax_output = input_list[1].detach()#detach用于阻断反向传播
    feature = input_list[0]
    if random_layer is None:
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))#torch.bmm（）用于将两个tensor相乘，且维度只能为三维
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))#将op_out输出为(不想计算,指定的)形状
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)))#size(1)返回random_out的第一维数据的个数
    batch_size = softmax_output.size(0) // 2
    # dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float()

    if entropy is not None:
        print("inside entropy")
        entropy.register_hook(grl_hook(coeff))#即对x求导时，对x的导数进行操作，并且register_hook的参数只能以函数的形式传过去。
        entropy = 1.0+torch.exp(-entropy)
        source_mask = torch.ones_like(entropy)
        source_mask[feature.size(0)//2:] = 0
        source_weight = entropy*source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[0:feature.size(0)//2] = 0
        target_weight = entropy*target_mask
        weight = source_weight / torch.sum(source_weight).detach().item() + \
                 target_weight / torch.sum(target_weight).detach().item()
        print(weight)
        return torch.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / torch.sum(weight).detach().item()
    else:
        return nn.BCELoss()(ad_out, dc_target)


def DANN(features, ad_net):
    ad_out = ad_net(features)
    batch_size = ad_out.size(0) // 2#整除向下取整
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float()
    return nn.BCELoss()(ad_out, dc_target)
