"""Dataset setting and data loader for MNIST."""


import torch
from torchvision import datasets, transforms

import params


def get_mnist(train):
    """Get MNIST dataset loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.ToTensor(),#灰度范围从0-255变成0-1
                                      # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),#增加这一行
                                      # transforms.Normalize(
                                      #     mean=params.dataset_mean,(0.5,),(0.5,)])
                                      #     std=params.dataset_std)])##R,G,B每层的归一化用到的均值和方差
                                      transforms.Normalize((0.5,),(0.5,))])

    # dataset and data loader
    mnist_dataset = datasets.MNIST(root=params.data_root,
                                   train=train,
                                   transform=pre_process,
                                   download=True)

    mnist_data_loader = torch.utils.data.DataLoader(
        dataset=mnist_dataset,#数据集来源
        batch_size=params.batch_size,
        shuffle=True)

    # print("#########################")
    # print(mnist_data_loader)#<torch.utils.data.dataloader.DataLoader object at 0x00000245AAC948E0>
    # print(type(mnist_data_loader))#<class 'torch.utils.data.dataloader.DataLoader'>
    return mnist_data_loader
