"""Test script to classify target data."""

import torch
import torch.nn as nn
import csv
from utils import make_variable


def eval_tgt(encoder, classifier, data_loader):
    """Evaluation for target encoder by source classifier on target dataset."""
    #用源分类器在目标数据集上评估目标编码器
    # set eval state for Dropout and BN layers


    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0
    acc = 0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    p = []#
    # evaluate network
    for (images, labels) in data_loader:
        # images = make_variable(images, volatile=True)

        with torch.no_grad():
            images = torch.autograd.Variable(images)
        labels = make_variable(labels).squeeze_()

        preds = classifier(encoder(images))
        loss += criterion(preds, labels).item()

        pred_cls = preds.data.max(1)[1]
        acc += pred_cls.eq(labels.data).cpu().sum()

        p.append(pred_cls)

    loss /= len(data_loader)
    # acc /= len(data_loader.dataset)
    acc = acc.float() / len(data_loader.dataset)

    print("############预测的标签############", p)
    print("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))
