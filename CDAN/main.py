#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#TODO: get curves
# TODO: try different combinations among datasets

from __future__ import division
import argparse
import warnings
from tqdm import tnrange
import torch
from torch.autograd import Variable
warnings.filterwarnings("ignore")

from train import train
from test import test
#from loss import CORAL_loss
from utils import load_pretrained_AlexNet, save_log, save_model, load_model
from dataloader import get_office_dataloader
from model import  AlexNet, AdversarialNetwork, baseNetwork
import network
import target as get_target_loader
import source as get_source_loader

# set model hyperparameters (paper page 5)
CUDA = True if torch.cuda.is_available() else False
CUDA = False
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
# BATCH_SIZE = [32, 32] # batch_s, batch_t [128, 56]
# EPOCHS = 1

def main():
    """
    This method puts all the modules together to train a neural network
    classifier using CORAL loss.

    Reference: https://arxiv.org/abs/1607.01719
    """
    parser = argparse.ArgumentParser(description="domain adaptation w CORAL")#创建一个解析对象

    parser.add_argument("--epochs", default=500, type=int,
                        help="number of training epochs")#每一个add_argument方法对应一个你要关注的参数或选项

    parser.add_argument("--batch_size_source", default=10, type=int,
                        help="batch size of source data")#help是用于注释

    parser.add_argument("--batch_size_target", default=10, type=int,
                        help="batch size of target data")#目标域的尺寸

    parser.add_argument("--name_source", default="source_data", type=str,
                        help="name of source dataset (default amazon)")

    parser.add_argument("--name_target", default="target_data", type=str,
                        help="name of source dataset (default webcam)")#目标域的名称

    parser.add_argument("--num_classes", default=5, type=int,
                        help="no. classes in dataset (default 5)")#分类的类别

    parser.add_argument("--load_model", default=None, type=None,
                        help="load pretrained model (default None)")


    args = parser.parse_args()#最后调用parse_args()方法进行解析

    # create dataloaders (Amazon as source and Webcam as target)
    print("creating source/target dataloaders...")
    print("source data:", args.name_source)#类似于调用arg函数里面的参数，name_source
    print("target data:", args.name_target)



    # source_loader = get_office_dataloader(name_dataset = args.name_source,
    #                                       batch_size = args.batch_size_source)
    #
    # target_loader = get_office_dataloader(name_dataset = args.name_target,
    #                                       batch_size = args.batch_size_target)

    src_data_loader, src_data_loader_eval = get_source_loader.source_loader(args.batch_size_source)
    tgt_data_loader, tgt_data_loader_eval = get_target_loader.target_loader(args.batch_size_target)


    ##<torch.utils.data.dataloader.DataLoader object at 0x000002382E5D4F40>##source_dataload
    ##<torch.utils.data.dataloader.DataLoader object at 0x000002382E5D4370>##target_dataload

    # define DeepCORAL network
    bottleneck_dim = 256 #256
    model = baseNetwork(num_classes=args.num_classes,bottleneck_dim=bottleneck_dim)#
    # model = network.AlexNetFc(use_bottleneck=True, bottleneck_dim=256, new_cls=True)
    ad_net = AdversarialNetwork(bottleneck_dim*args.num_classes,1024)#输入特征为256x分类的类别，hiddensize = 1024
    model.train(True)
    ad_net.train(True)
    # define optimizer pytorch: https://pytorch.org/docs/stable/optim.html
    # specify learning rates per layers:
    # 10*learning_rate for last two fc layers according to paper
    optimizer = torch.optim.SGD([
        {"params": model.sharedNetwork.parameters()},
        {"params": model.fc8.parameters(), "lr":10*LEARNING_RATE},
        {"params":ad_net.parameters(), "lr_mult": 10, 'decay_mult': 2}
    ], lr=LEARNING_RATE, momentum=MOMENTUM)
    # optimizer = torch.optim.SGD([
    #     {"params": model.sharedNetwork.parameters()},
    #     {"params": model.fc8.parameters(), "lr":10*LEARNING_RATE},
    # ], lr=LEARNING_RATE, momentum=MOMENTUM)


    # move to CUDA if available
    if CUDA:
        model = model.cuda()
        print("using cuda...")

    # load pre-trained model or pre-trained AlexNet
    # if args.load_model is not None:
    #     load_model(model, args.load_model) # contains path to model params
    # else:
    #     load_pretrained_AlexNet(model.sharedNetwork, progress=True)

    print("model type:", type(model))#<class 'model.baseNetwork'>

    # store statistics of train/test
    training_s_statistic = []
    testing_s_statistic = []
    testing_t_statistic = []

    # start training over epochs
    print("running training for {} epochs...".format(args.epochs))#打印训练模型的迭代次数
    for epoch in range(0, args.epochs):#从0到10
        # compute lambda value from paper (eq 6)
        lambda_factor = (epoch+1)/args.epochs

        # run batch trainig at each epoch (returns dictionary with epoch result)
        result_train = train(model, ad_net, src_data_loader, tgt_data_loader,
                             optimizer, epoch+1, lambda_factor, CUDA)


        # print(result_train)#打印结果
        # print(type(result_train))#列表
        # print log values
        ########在训练的一代中计算所有步的平均loss##########
        ########result——train得到的是列表，每个元素是一个dict############
        print("[EPOCH] {}: Classification: {:.6f}, CDAN loss: {:.6f}, Total_Loss: {:.6f}".format(
                epoch+1,
                sum(row['classification_loss'] / row['total_steps'] for row in result_train),
                sum(row['cdan_loss'] / row['total_steps'] for row in result_train),
                sum(row['total_loss'] / row['total_steps'] for row in result_train),
            ))


        training_s_statistic.append(result_train)#将源域测试的结果存入

        # perform testing simultaneously: classification accuracy on both dataset
        test_source = test(model, src_data_loader_eval, epoch, CUDA)
        test_target = test(model, tgt_data_loader_eval, epoch, CUDA)
        testing_s_statistic.append(test_source)
        testing_t_statistic.append(test_target)

        print("[Test Source]: Epoch: {}, avg_loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
                epoch+1,
                test_source['average_loss'],
                test_source['correct_class'],
                test_source['total_elems'],
                test_source['accuracy %'],
            ))

        print("[Test Target]: Epoch: {}, avg_loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
                epoch+1,
                test_target['average_loss'],
                test_target['correct_class'],
                test_target['total_elems'],
                test_target['accuracy %'],
        ))

    # save results
    print("saving results...")
    # save_log(training_s_statistic, 'CDAN_amz_dslr/no_adaptation_training_s_statistic.pkl')
    # save_log(testing_s_statistic, 'CDAN_amz_dslr/no_adaptation_testing_s_statistic.pkl')
    # save_log(testing_t_statistic, 'CDAN_amz_dslr/no_adaptation_testing_t_statistic.pkl')
    # #save_log(training_s_statistic, 'CDAN_amz_dslr/training_s_statistic.pkl')##
    # #save_log(testing_s_statistic, 'CDAN_amz_dslr/testing_s_statistic.pkl')##
    # #save_log(testing_t_statistic, 'CDAN_amz_dslr/testing_t_statistic.pkl')##

    save_log(training_s_statistic, 'cdan_model/training_s_statistic.pkl')
    save_log(testing_s_statistic, 'cdan_model/testing_s_statistic.pkl')
    save_log(testing_t_statistic, 'cdan_model/testing_t_statistic.pkl')

    save_model(model, 'checkpoint.tar')


if __name__ == '__main__':
    main()
