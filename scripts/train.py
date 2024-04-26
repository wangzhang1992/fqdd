import torch
import os
# import copy, math
import numpy as np
import model
import blstm as blstm
import cnn_dnn_ctc
import torch.optim.lr_scheduler as lr_sch
import torch.nn.functional as F
#import matplotlib.pyplot as plt
#import matplotlib
from lang import dicts, readlexicon
from DataLoader import My_Data
from argument import parse_arguments
from Levenshtein import distance
from warpctc_pytorch import CTCLoss
from decode import int2word, GreedyDecoder

#matplotlib.use('Qt5Agg')
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# 第epoch值进行计算并更新学习率

train_cer = []
train_loss = []
dev_loss = []
dev_cer = []
lr_list = []
flag_plot = True

def adjust_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    lr_list.append(lr)


def calculate_loss(ctc_loss, pred, gold, input_lengths, target_lengths):
    """
    Calculate loss
    args:
        pred: B x T x C
        gold: B x T
        input_lengths: B (for CTC)
        target_lengths: B (for CTC)
        smoothing:
        type: ce|ctc (ctc => pytorch 1.0.0 or later)
        input_lengths: B (only for ctc)
        target_lengths: B (only for ctc)
    """
    log_probs = pred.transpose(0, 1)  # T x B x C
    # print(gold.size())
    targets = gold
    #targets = gold.contiguous().view(-1)  # (B*T)

    """
    log_probs: torch.Size([209, 8, 3793])
    targets: torch.Size([8, 46])
    input_lengths: torch.Size([8])
    target_lengths: torch.Size([8])
    """

    #log_probs = F.log_softmax(log_probs, dim=2)
    #log_probs = log_probs.detach().requires_grad_()
    loss = ctc_loss(log_probs.to("cpu"), targets.to("cpu"), input_lengths.to("cpu"), target_lengths.to("cpu"))

    return loss


def train(asr_dnn, train_load, test_load, optimizer, scheduler_l, ctc_loss, args, dic, device):
    min_loss = 1000
    epoch_n = args.epoch_n
    asr_dnn_save_path = args.model_save_path
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)

    for epoch in range(1,epoch_n):

        asr_dnn.train()
        print("Epoch {}/{}".format(epoch, epoch_n))
        print("-" * 10)
        loss100 = 0
        acc100 = 0
        for idx, data in enumerate(train_load):
            running_acc = 0
            running_loss = 0
            X_train, Y_train, wavlist_length, label_length = data


            # wavlist_length:tensor([110, 111, 166, 280, 158, 223, 321, 107, 293,  33, 238,  98, 182, 202,180, 278])
            # label_length:tensor([16, 17, 21, 39, 23, 30, 44, 13, 41,  4, 30, 11, 23, 29, 26, 42])
            X_train = torch.FloatTensor(X_train).to(device)
            Y_train = np.hstack([Y_train[i, :j] for i, j in enumerate(label_length)])
            Y_train = torch.IntTensor(Y_train)
            wavlist_length = torch.IntTensor(wavlist_length)
            label_length = torch.IntTensor(label_length)

            output = asr_dnn(X_train)
            optimizer.zero_grad()
            loss = calculate_loss(ctc_loss, output, Y_train, wavlist_length, label_length)
            loss.backward()
            torch.nn.utils.clip_grad_norm(asr_dnn.parameters(), 500)
            optimizer.step()

            running_loss = loss.item() / args.batch_size

            predword, targ = GreedyDecoder(output, Y_train, wavlist_length, label_length, dic)
            for i in range(len(targ)):
                running_acc += distance(''.join(targ[i]), ''.join(predword[i])) / label_length[i]

            running_acc = 1 - running_acc / args.batch_size
            loss100 += running_loss
            acc100 += running_acc
            if idx % 100 == 0 and idx != 0:
                print("Train:\t\tEpoch:{}\tbatch:{}\tloss:{}\tacc:{}".format(
                    epoch, idx, loss100/100, acc100/100))
                train_cer.append(acc100/100)
                train_loss.append(loss100/100)

                loss100 = 0
                acc100 = 0

        dloss, dacc = test_asr_dnn(asr_dnn, test_load, ctc_loss, args, dic, device)
        scheduler_l.step(dloss)
        if min_loss > dloss:
            torch.save(asr_dnn.state_dict(), asr_dnn_save_path + '/asr_dnn_{}.pth'.format(epoch),
                       _use_new_zipfile_serialization=False)
            min_loss = dloss
        else:
            print("jump")
            #adjust_lr(optimizer, optimizer.state_dict()['param_groups'][0]['lr'] / 2)
            #adjust_lr(optimizer, optimizer.state_dict()['param_groups'][0]['lr'] / 2)
        print("SAVE DEV :Loss:{:.4f}\tDEV Accuracy is:{:.4f}%\tlr:{}"
                  .format(dloss, dacc,optimizer.state_dict()['param_groups'][0]['lr']))


def test_asr_dnn(asr_dnn, test_load, ctc_loss, args, dic, device):

    asr_dnn.eval()
    dloss = 0
    trunning_acc = 0
    loss100 = 0
    acc100 = 0
    for idx, data in enumerate(test_load):
        running_acc = 0.
        running_loss = 0.

        X_test, Y_test, wavlist_length, label_length = data

        X_test = torch.FloatTensor(X_test).to(device)
        Y_test = np.hstack([Y_test[i, :j] for i, j in enumerate(label_length)])
        Y_test = torch.IntTensor(Y_test)
        wavlist_length = torch.IntTensor(wavlist_length)
        label_length = torch.IntTensor(label_length)

        toutput = asr_dnn(X_test)
        loss = calculate_loss(ctc_loss, toutput, Y_test, wavlist_length, label_length)
        # print("touput.shape:{}\ntoutput.data={}".format(toutput.shape,toutput))
        tpredword, ttarg = GreedyDecoder(toutput, Y_test, wavlist_length, label_length, dic)
        for i in range(len(tpredword)):
            print("pred:{}\ttarg:{}".format(''.join(tpredword[i]), ''.join(ttarg[i])))
            running_acc += distance(''.join(ttarg[i]), ''.join(tpredword[i])) / label_length[i]

        running_acc = 1 - running_acc / args.batch_size
        running_loss = loss.item() / args.batch_size

        loss100 += running_loss
        acc100 += running_acc
        trunning_acc += running_acc
        dloss += running_loss

        if idx % 100 == 0 and idx != 0:
            dev_cer.append(acc100/100)
            dev_loss.append(loss100/100)
            loss100 = 0
            acc100 = 0

    return dloss / len(test_load), trunning_acc / (len(test_load))


def visualization_of_deep_learning_training(batch):

    xx = [i*100*batch for i in range(len(train_loss))]
    dx = [i*100*batch for i in range(len(dev_loss))]

    fig = plt.figure()

    trloss = fig.add_subplot(231)
    trloss.set(xlim=[0, len(train_loss)*100 + 20], title="train_loss",
               ylabel='loss', xlabel='batch')
    trloss.plot(xx, train_loss, color='darkred')

    devloss = fig.add_subplot(232)
    devloss.set(xlim=[0, len(dev_loss)*100 + 20], title="dev_loss",
                ylabel='loss', xlabel='batch')
    devloss.plot(dx, dev_loss, color='g')

    trcer = fig.add_subplot(233)
    trcer.set(xlim=[0, len(train_loss)*100 + 20], ylim=[0, 1], title="train_cer",
              ylabel='acc', xlabel='batch')
    trcer.plot(xx, train_cer, color='red')

    dcer = fig.add_subplot(234)
    dcer.set(xlim=[0, len(dev_loss)*100 + 20], ylim=[0, 1], title="dev_cer",
             ylabel='acc', xlabel='batch')
    dcer.plot(dx, dev_cer, color='gray')

    plt.show()


def model_init(asr_dnn):
    for p in asr_dnn.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform(p)

def main():
    torch.manual_seed(2021)
    torch.cuda.manual_seed_all(2021)
    np.random.seed(2021)
    args = parse_arguments()
    os.system('. ./path.sh')
    #os.system('feats_extract.sh raw_data/')

    if os.path.exists('data/phones.txt'):
        with open('data/phones.txt', 'r', encoding='utf-8') as rf:
            dic = {}
            for line in rf:
                lines = line.strip().split(' ')
                dic[lines[0]] = int(lines[1])
        #print("202:{}".format(dic))

    if args.label_status == 'phones':
        if not os.path.exists('data/phones.txt'):
            dic = dicts(args.train_path, args.test_path, args.dev_path, args.lexicon_path)
        train_load = My_Data(mode='train', batch_size=args.batch_size, lexicon=args.lexicon_path, phones=dic)
        test_load = My_Data(mode='test', batch_size=args.batch_size, lexicon=args.lexicon_path, phones=dic)
        # dev_load = My_Data(mode='dev', batch_size=args.batch_size, lexicon=args.lexicon_path, phones = dic)
        #print("214:{}".format(dic))
    else:
        if not os.path.exists('data/phones.txt'):
            dic = dicts(args.train_path, args.test_path, args.dev_path)
            #print("218:{}".format(dic))
        train_load = My_Data(mode='train', batch_size=args.batch_size, phones=dic)
        test_load = My_Data(mode='test', batch_size=args.batch_size, phones=dic)
        # dev_load = My_Data(mode='dev', batch_size=args.batch_size, phones=dic)

    input_dim = int(open('mfcc_pitch/feat_dim', 'r').readline().strip())
    output_dim = len(dic)
    ctc_loss = CTCLoss()

    #asr_dnn = model.Model(input_dim, output_dim)

    #asr_dnn = cnn_dnn_ctc.CNN(output_dim)
    # input_dim, output_dim, hiddle, layers, drop, bi
    asr_dnn = blstm.BlstmNet(input_dim, output_dim, 512, 3, 0, True)
    model_init(asr_dnn)
    print(asr_dnn)

    device = torch.device('cuda:0')
    asr_dnn.to(device)
    # SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True)
    optimizer = torch.optim.Adam(asr_dnn.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    # optimizer = torch.optim.SGD(asr_dnn.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    #optimizer = torch.optim.SGD(asr_dnn.parameters(), lr=args.lr, momentum=0.9)
    #scheduler_l = lr_sch.LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(args.epoch_n+1))
    scheduler_l = lr_sch.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=False,
                                           threshold=0.0001, threshold_mode='rel', min_lr=0, cooldown=0, eps=1e-08)

    train(asr_dnn, train_load, test_load, optimizer, scheduler_l, ctc_loss, args, dic, device)
    visualization_of_deep_learning_training(args.batch_size)


if __name__ == "__main__":

    main()
