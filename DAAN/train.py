from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import math
import data_loader
from model import DAAN
from torch.utils import model_zoo
import numpy as np
from IPython import embed
import tqdm

# Training settings
parser = argparse.ArgumentParser(description='PyTorch DAAN')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')#训练的输入尺寸
parser.add_argument('--epochs', type=int, default=1, metavar='N',#200
                    help='number of epochs to train (default: 10)')#训练的代数
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',#学习率
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',#动量
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,#是否采用gpu
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=233, metavar='S',#固定随机数的种子，用于复现模型
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',#10日志间隔
                    help='how many batches to wait before logging training status')
parser.add_argument('--l2_decay', type=float, default=5e-4,
                    help='the L2  weight decay')
parser.add_argument('--save_path', type=str, default="./tmp/origin_",
                    help='the path to save the model')#保存模型的路径
parser.add_argument('--root_path', type=str, default="officehome/",#加载数据集的路径
                    help='the path to load the data')
parser.add_argument('--source_dir', type=str, default="Clipart",#源域数据集的名称
                    help='the name of the source dir')
parser.add_argument('--test_dir', type=str, default="Product",#目标域数据集的路径
                    help='the name of the test dir')
parser.add_argument('--diff_lr', type=bool, default=True,
                    help='the fc layer and the sharenet have different or same learning rate')
parser.add_argument('--gamma', type=int, default=1,
                    help='the fc layer and the sharenet have different or same learning rate')
parser.add_argument('--num_class', default=65, type=int,
                    help='the number of classes')#分类的类别
parser.add_argument('--gpu', default=0, type=int)

args = parser.parse_args()#将上述添加 到parser的项值联合到一起
args.cuda = not args.no_cuda and torch.cuda.is_available()
DEVICE = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
torch.cuda.manual_seed(args.seed)
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

def load_data():
    source_train_loader = data_loader.load_training(args.root_path, args.source_dir, args.batch_size, kwargs)
    target_train_loader = data_loader.load_training(args.root_path, args.test_dir, args.batch_size, kwargs)
    target_test_loader  = data_loader.load_testing(args.root_path, args.test_dir, args.batch_size, kwargs)

    #打印输入数据的形式#
    # print(source_train_loader, target_train_loader, target_test_loader)
    #<torch.utils.data.dataloader.DataLoader object at 0x000001EE7FF4BDF0>#
    # <torch.utils.data.dataloader.DataLoader object at 0x000001EE083CD100>#
    # <torch.utils.data.dataloader.DataLoader object at 0x000001EE083CD3D0>#
    return source_train_loader, target_train_loader, target_test_loader


def print_learning_rate(optimizer):
    for p in optimizer.param_groups:
        outputs = ''
        for k, v in p.items():
            if k == 'params':# ==与is
                outputs += (k + ': ' + str(v[0].shape).ljust(30) + ' ')
            else:
                outputs += (k + ': ' + str(v).ljust(10) + ' ')

        # print("##打印optimizer的output##")
        # print(type(outputs))##<class 'str'>

        print(outputs)



def train(epoch, model, source_loader, target_loader):
    #total_progress_bar = tqdm.tqdm(desc='Train iter', total=args.epochs)
    LEARNING_RATE = args.lr / math.pow((1 + 10 * (epoch - 1) / args.epochs), 0.75)#Math.pow(底数,几次方)
    if args.diff_lr:
        optimizer = torch.optim.SGD([
            {'params': model.sharedNet.parameters()},
            {'params': model.bottleneck.parameters()},
            {'params': model.domain_classifier.parameters()},
            {'params': model.dcis.parameters()},
            {'params': model.source_fc.parameters(), 'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE / 10, momentum=args.momentum, weight_decay=args.l2_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=args.momentum,weight_decay = args.l2_decay)

    print_learning_rate(optimizer)#调用学习率函数

    global D_M, D_C, MU#声明为全局变量
    model.train()
    len_dataloader = len(source_loader)
    DEV = DEVICE

    d_m = 0
    d_c = 0
    ''' update mu per epoch '''#每一代都更新mu
    if D_M==0 and D_C==0 and MU==0:
        MU = 0.5
    else:
        D_M = D_M/len_dataloader
        D_C = D_C/len_dataloader
        MU = 1 - D_M/(D_M + D_C)

    ######################################
    #Tqdm 是一个快速，可扩展的Python进度条，#
    #可以在 Python 长循环中添加一个进度提示信息，#
    #用户只需要封装任意的迭代器 tqdm(iterator)#
    #####################################
    for batch_idx, (source_data, source_label) in tqdm.tqdm(enumerate(source_loader),
                                    total=len_dataloader,
                                    desc='Train epoch = {}'.format(epoch), ncols=80, leave=False):
        p = float(batch_idx+1 + epoch * len_dataloader) / args.epochs / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        optimizer.zero_grad()
        source_data, source_label = source_data.to(DEVICE), source_label.to(DEVICE)
        for target_data, target_label in target_loader:
            target_data, target_label = target_data.to(DEVICE), target_label.to(DEVICE)#是否需要将数据放到gpu上面运行
            break
        out = model(source_data, target_data, source_label, DEV, alpha)# source, s_domain_output, t_domain_output, s_out, t_out
        s_output, s_domain_output, t_domain_output = out[0],out[1],out[2]
        s_out = out[3]
        t_out = out[4]

        #将源域的域标签定义为0，目标域的标签定义为1
        #global loss
        sdomain_label = torch.zeros(args.batch_size).long().to(DEV)#返回一个形状为为size,类型为torch.dtype，里面的每一个值都是0的tensor
        err_s_domain = F.nll_loss(F.log_softmax(s_domain_output, dim=1), sdomain_label)#NLLLoss 的 输入 是一个对数概率向量和一个目标标签. 它不会为我们计算对数概率. 适合网络的最后一层是log_softmax. 损失函数
        tdomain_label = torch.ones(args.batch_size).long().to(DEV)
        err_t_domain = F.nll_loss(F.log_softmax(t_domain_output, dim=1), tdomain_label)

        #local loss
        loss_s = 0.0
        loss_t = 0.0
        tmpd_c = 0
        for i in range(args.num_class):
            loss_si = F.nll_loss(F.log_softmax(s_out[i], dim=1), sdomain_label)
            loss_ti = F.nll_loss(F.log_softmax(t_out[i], dim=1), tdomain_label)
            loss_s += loss_si
            loss_t += loss_ti
            tmpd_c += 2 * (1 - 2 * (loss_si + loss_ti))
        tmpd_c /= args.num_class

        d_c = d_c + tmpd_c.cpu().item()#显示精度上的区别，所以在求loss,以及accuracy rate的时候一般用item()而不是[1,1]

        global_loss = 0.05*(err_s_domain + err_t_domain)
        local_loss = 0.01*(loss_s + loss_t)

        d_m = d_m + 2 * (1 - 2 * global_loss.cpu().item())

        join_loss = (1 - MU) * global_loss + MU * local_loss
        soft_loss = F.nll_loss(F.log_softmax(s_output, dim=1), source_label)
        if args.gamma == 1:
            gamma = 2 / (1 + math.exp(-10 * (epoch) / args.epochs)) - 1
        if args.gamma == 2:
            gamma = epoch /args.epochs
        loss = soft_loss + join_loss
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:#打印步骤信息，每10步
            print('\nLoss: {:.6f},  label_Loss: {:.6f},  join_Loss: {:.6f}, global_Loss:{:.4f}, local_Loss:{:.4f}'.format(
                loss.item(), soft_loss.item(), join_loss.item(), global_loss.item(), local_loss.item()))
        #total_progress_bar.update(1)
    D_M = np.copy(d_m).item()
    D_C = np.copy(d_c).item()

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    # 在使用pytorch时，并不是所有的操作都需要进行计算图的生成（计算过程的构建，以便梯度反向传播等操作）。
    # 而对于tensor的计算操作，默认是要进行计算图的构建的，在这种情况下，可以使用 with torch.no_grad():，
    # 强制之后的内容不进行计算图构建。就是说使用with.no_grad()得到的结果并不能进行梯度反转计算
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            out = model(data, data, target, DEVICE)
            s_output = out[0]#预测的标签
            test_loss += F.nll_loss(F.log_softmax(s_output, dim = 1), target, size_average=False).item() # sum up batch loss
            pred = s_output.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()#比较预测标签的索引位置和真实标签的相比较，得到的标签相同的个数

            # print('打印数据',data)#, 0.0000]]]])
            # print('标签', target)#tensor([48,
            # print('打印s_output的类别',s_output)#tensor([[ 0.2145,  0.53
            # print(type(s_output))#<class 'torch.Tensor'>


        test_loss /= len(test_loader.dataset)
        print(args.test_dir, '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    return correct  #返回预测正确的标签个数


if __name__ == '__main__':
    model = DAAN.DAANNet(num_classes=args.num_class, base_net='ResNet50').to(DEVICE)
    print(model)
    train_loader, unsuptrain_loader, test_loader = load_data()#调用load_data函数
    correct = 0
    D_M = 0
    D_C = 0
    MU = 0
    for epoch in range(1, args.epochs + 1):
        train_loader, unsuptrain_loader, test_loader = load_data()
        train(epoch, model, train_loader, unsuptrain_loader)
        t_correct = test(model, test_loader)
        if t_correct > correct:
            correct = t_correct
        print("%s max correct:" % args.test_dir, correct)
        print(args.source_dir, "to", args.test_dir)
