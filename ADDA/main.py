"""Main script for ADDA."""

import params
from core import eval_src, eval_tgt, train_src, train_tgt
from models import Discriminator, LeNetClassifier, LeNetEncoder
from utils import get_data_loader, init_model, init_random_seed
import get_test_data as get_test
import get_train_data as get_train


if __name__ == '__main__':
    # init random seed
    init_random_seed(params.manual_seed)#用于复现模型torch.manual_seed(1)

    # load dataset
    src_data_loader = get_data_loader(params.src_dataset)#加载源域的训练集
    src_data_loader_eval = get_data_loader(params.src_dataset, train=False)#加载源域的测试集

    # print(src_data_loader)<torch.utils.data.dataloader.DataLoader object at 0x0000027B7A7047C0>
    # print(src_data_loader_eval)<torch.utils.data.dataloader.DataLoader object at 0x0000027B010158E0>

    # tgt_data_loader = get_data_loader(params.tgt_dataset)#加载Usps数据集，如果pycharm因为http访问问题不能正常下载，则可以手动从官网上下载
    # tgt_data_loader_eval = get_data_loader(params.tgt_dataset, train=False)

    tgt_data_loader = get_train.get_train_data(50)  # 加载Usps数据集，如果pycharm因为http访问问题不能正常下载，则可以手动从官网上下载
    tgt_data_loader_eval = get_test.get_test_data(50)

    # load models
    src_encoder = init_model(net=LeNetEncoder(),
                             restore=params.src_encoder_restore)
    src_classifier = init_model(net=LeNetClassifier(),#调用
                                restore=params.src_classifier_restore)#src_classifier_restore = "snapshots/ADDA-source-classifier-final.pt"
    tgt_encoder = init_model(net=LeNetEncoder(),#调用lenet.py
                             restore=params.tgt_encoder_restore)#tgt_encoder_restore ="snapshots/ADDA-target-encoder-final.pt"
    critic = init_model(Discriminator(input_dims=params.d_input_dims,#500
                                      hidden_dims=params.d_hidden_dims,#500
                                      output_dims=params.d_output_dims),#2
                        restore=params.d_model_restore)#d_model_restore = "snapshots/ADDA-critic-final.pt"

    # train source model
    print("=== Training classifier for source domain ===")
    print(">>> Source Encoder <<<")
    print(src_encoder)
    print(">>> Source Classifier <<<")
    print(src_classifier)

    if not (src_encoder.restored and src_classifier.restored and
            params.src_model_trained):
        src_encoder, src_classifier = train_src(
            src_encoder, src_classifier, src_data_loader)

    # eval source model
    print("=== Evaluating classifier for source domain ===")
    eval_src(src_encoder, src_classifier, src_data_loader_eval)

    # train target encoder by GAN
    print("=== Training encoder for target domain ===")
    print(">>> Target Encoder <<<")
    print(tgt_encoder)
    print(">>> Critic <<<")
    print(critic)

    # init weights of target encoder with those of source encoder
    # if not tgt_encoder.restored:
    #     tgt_encoder.load_state_dict(src_encoder.state_dict())

    if not (tgt_encoder.restored and critic.restored and
            params.tgt_model_trained):
        tgt_encoder = train_tgt(src_encoder, tgt_encoder, critic,
                                src_data_loader, tgt_data_loader)

    # eval target encoder on test set of target dataset
    print("=== Evaluating classifier for encoded target domain ===")
    print(">>> source only <<<")
    eval_tgt(src_encoder, src_classifier, tgt_data_loader_eval)


#############新增标签####@##########
    label_t = []
    for i, (x,y) in enumerate(tgt_data_loader_eval):
        label_t.append(y)

    print('############实际的标签值#############', label_t)
###################################


    print(">>> domain adaption <<<")
    eval_tgt(tgt_encoder, src_classifier, tgt_data_loader_eval)
