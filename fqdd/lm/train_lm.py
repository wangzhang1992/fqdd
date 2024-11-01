import torch
import os
from script.tools.argument import parse_arguments
import numpy as np
import torch.optim.lr_scheduler as lr_sch
from script.tools.lang import create_phones, read_phones
from script.lm.lm_net import RNN_Model
from script.tools.load_data import load_txt_data
from script.asr.decode import int2word
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Qt5Agg')
train_loss = []
dev_loss = []

flag_plot = True


def train(lm_mode, train_load, dev_load, optimizer, scheduler_l, ce_loss, args, start_epoch, dic, device):
    min_loss = 1000
    epoch_n = args.epoch_n
    asr_dnn_save_path = args.model_save_path

    for epoch in range(start_epoch, epoch_n):

        lm_mode.train()
        print("Epoch {}/{}".format(epoch, epoch_n))
        print("-" * 10)

        tloss = 0.

        for idx, data in enumerate(train_load):
            x_train, label_length = data
            x_train = torch.tensor(x_train, dtype=torch.long)
            x_train = x_train.to(device)
            pre = lm_mode(x_train)
            res = torch.argmax(pre, dim=2)
            # print([item for item in int2word(res[0], dic) if item != '<ese>'])
            # print([item for item in int2word(x_train[0], dic) if item != '<ese>'])
            pre = torch.transpose(pre, 1, 2)
            optimizer.zero_grad()
            loss = ce_loss(pre, x_train)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lm_mode.parameters(), 500)
            optimizer.step()

            print("Train:\t\tEpoch:{}\tbatch:{}\tloss:{:.2f}".format(
                epoch, idx, loss.item()))

            tloss += loss.item()
        print("Loss:{:.2f}".format(tloss / len(train_load)))
        train_loss.append(tloss / len(train_load))

        dloss = test(lm_mode, dev_load, ce_loss, args, dic, device)

        scheduler_l.step(dloss)
        if min_loss > dloss:
            checkpoint = {
                'model': lm_mode.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint, asr_dnn_save_path + '/lm_model_{}.pt'.format(epoch),
                       _use_new_zipfile_serialization=False)
            min_loss = dloss
        else:
            print("jump")
            # adjust_lr(optimizer, optimizer.state_dict()['param_groups'][0]['lr'] / 2)
            # adjust_lr(optimizer, optimizer.state_dict()['param_groups'][0]['lr'] / 2)
        print("Loss:{:.2f}\tlr:{}"
              .format(dloss, optimizer.state_dict()['param_groups'][0]['lr']))


def test(lm_mode, test_load, ce_loss, args, dic, device):
    lm_mode.eval()
    dloss = 0

    for idx, data in enumerate(test_load):
        x_test, label_length = data
        x_test = x_test.to(device)
        x_test = torch.tensor(x_test, dtype=torch.long)
        pre = lm_mode(x_test)
        pred = torch.transpose(pre, 1, 2)
        loss = ce_loss(pred, x_test)
        dloss += loss.item()
        res = torch.argmax(pre, dim=2)
        # print([item for item in int2word(res[0], dic) if item != '<ese>'])
        # print([item for item in int2word(x_test[0], dic) if item != '<ese>'])
    dev_loss.append(dloss / len(test_load))

    return dloss / len(test_load)


def visualization_of_deep_learning_training(start_epoch, epoch_n):
    xx = [i for i in range(start_epoch, epoch_n)]

    fig = plt.figure()

    trloss = fig.add_subplot(121)
    trloss.set(xlim=[0, epoch_n + 20],
               ylim=[np.min(train_loss) - 5 if np.min(train_loss) < 0. else 0, np.max(train_loss) + 20],
               title="train_loss",
               ylabel="loss", xlabel='epoch')
    trloss.plot(xx, train_loss, color='darkred')

    devloss = fig.add_subplot(122)
    devloss.set(xlim=[0, epoch_n + 20],
                ylim=[np.min(dev_loss) - 5 if np.min(dev_loss) < 0. else 0, np.max(dev_loss) + 20],
                title="dev_loss",
                ylabel="loss", xlabel='epoch')
    devloss.plot(xx, dev_loss, color='g')

    plt.show()


def model_init(model):
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)


def main():
    args = parse_arguments()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    if os.path.exists(args.phones_path):
        dic = read_phones(args.phones_path)
    else:
        dic = create_phones(args)

    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
    ce_loss = torch.nn.CrossEntropyLoss()
    print(len(dic))
    lm_model = RNN_Model(len(dic), args)
    model_init(lm_model)
    start_epoch = 1
    lm_model.to(device)

    optimizer = torch.optim.Adam(lm_model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    # optimizer = torch.optim.SGD(asr_dnn.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    # optimizer = torch.optim.SGD(asr_dnn.parameters(), lr=args.lr, momentum=0.9)
    # scheduler_l = lr_sch.LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(args.epoch_n+1))
    scheduler_l = lr_sch.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, verbose=False,
                                           threshold=0.0001, threshold_mode='rel', min_lr=0, cooldown=0, eps=1e-08)
    # load training data
    train_load = load_txt_data(args.feat_save_path + '/train', args, shuffle=True)
    dev_load = load_txt_data(args.feat_save_path + '/dev', args, shuffle=False)
    # test_load = load_data(args.test_path+'/test', args, shuffle=False)

    train(lm_model, train_load, dev_load, optimizer, scheduler_l, ce_loss, args, start_epoch, dic, device)
    visualization_of_deep_learning_training(start_epoch, args.epoch_n)


if __name__ == "__main__":
    main()
