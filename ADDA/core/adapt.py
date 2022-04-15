"""Adversarial adaptation to train target encoder."""

import os

import torch
import torch.optim as optim
from torch import nn

import params
from utils import make_variable


def train_tgt(src_encoder, tgt_encoder, critic,
              src_data_loader, tgt_data_loader):
    """Train encoder for target domain."""
    ####################
    # 1. setup network #
    ####################

    # set train state for Dropout and BN layers
    tgt_encoder.train()
    critic.train()

    # setup criterion and optimizer
    criterion = nn.CrossEntropyLoss()#计算交叉熵损失
    optimizer_tgt = optim.Adam(tgt_encoder.parameters(),
                               lr=params.c_learning_rate,
                               betas=(params.beta1, params.beta2))#beta1 = 0.5 beta2 = 0.9
    optimizer_critic = optim.Adam(critic.parameters(),
                                  lr=params.d_learning_rate,#d_learning_rate = 1e-4
                                  betas=(params.beta1, params.beta2))#beta1 = 0.5 beta2 = 0.9
    len_data_loader = min(len(src_data_loader), len(tgt_data_loader))#加载的数据的长度取两者之间最小的

    ####################
    # 2. train network #
    # ###################
    # print("#############到了训练目标域编码器######")

    for epoch in range(params.num_epochs):#num_epochs = 2000#
        # zip source and target data pair
        data_zip = enumerate(zip(src_data_loader, tgt_data_loader))#zip函数对象中对应的元素打包成一个个元组，然后返回由这些元组组成的对象
        for step, ((images_src, _), (images_tgt, _)) in data_zip:
            ###########################
            # 2.1 train discriminator #
            ###########################

            # make images variable
            images_src = make_variable(images_src)
            images_tgt = make_variable(images_tgt)

            # zero gradients for optimizer
            optimizer_critic.zero_grad()

            # extract and concat features（提取和拼接特征）
            feat_src = src_encoder(images_src)#源域特征
            feat_tgt = tgt_encoder(images_tgt)#目标域特征
            feat_concat = torch.cat((feat_src, feat_tgt), 0)
            # print("到了拼接源域和目标域特征")

            # predict on discriminator
            pred_concat = critic(feat_concat.detach())#辨别器预测的标签

            # prepare real and fake label
            label_src = make_variable(torch.ones(feat_src.size(0)).long())#源域的标签
            label_tgt = make_variable(torch.zeros(feat_tgt.size(0)).long())#目标域的标签
            label_concat = torch.cat((label_src, label_tgt), 0)#特征组合后的标签

            # compute loss for critic
            loss_critic = criterion(pred_concat, label_concat)#计算cross-entropy
            loss_critic.backward()#反向传播

            # optimize critic
            optimizer_critic.step()

            pred_cls = torch.squeeze(pred_concat.max(1)[1])#独热编码转化
            acc = (pred_cls == label_concat).float().mean()#通过布尔值求平均准确率

            ############################
            # 2.2 train target encoder #
            ############################



            # zero gradients for optimizer
            optimizer_critic.zero_grad()
            optimizer_tgt.zero_grad()

            # extract and target features
            feat_tgt = tgt_encoder(images_tgt)

            # predict on discriminator
            pred_tgt = critic(feat_tgt)

            # prepare fake labels
            label_tgt = make_variable(torch.ones(feat_tgt.size(0)).long())

            # compute loss for target encoder
            loss_tgt = criterion(pred_tgt, label_tgt)
            loss_tgt.backward()

            # optimize target encoder
            optimizer_tgt.step()

            #######################
            # 2.3 print step info #打印步骤信息
            #######################
            if ((step + 1) % params.log_step == 0):#log_step = 100
                print("Epoch [{}/{}] Step [{}/{}]:"
                      "d_loss={:.5f} g_loss={:.5f} acc={:.5f}"
                      .format(epoch + 1,
                              params.num_epochs,#save_step = 100
                              step + 1,
                              len_data_loader,
                              loss_critic.item(),
                              loss_tgt.item(),
                              acc.item()))

        #############################
        # 2.4 save model parameters #
        #############################
        if ((epoch + 1) % params.save_step == 0):#save_step_pre = 100
            torch.save(critic.state_dict(), os.path.join(
                params.model_root,
                "ADDA-critic-{}.pt".format(epoch + 1)))
            torch.save(tgt_encoder.state_dict(), os.path.join(
                params.model_root,
                "ADDA-target-encoder-{}.pt".format(epoch + 1)))

    torch.save(critic.state_dict(), os.path.join(
        params.model_root,
        "ADDA-critic-final.pt"))
    torch.save(tgt_encoder.state_dict(), os.path.join(
        params.model_root,
        "ADDA-target-encoder-final.pt"))

    # print("tgt_encoder训练好了")
    return tgt_encoder
