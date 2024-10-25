import torch
import os,sys
import numpy as np
sys.path.insert(0, "./")
import torch.optim.lr_scheduler as lr_sch
import matplotlib.pyplot as plt
import matplotlib
# from apex import amp
from script.utils.feature import get_feats
from script.prepare_data.aishell_prepare_data import prepare_data
from script.utils.lang import create_phones, read_phones
from script.utils.load_data import load_data_v2
from script.utils.argument import parse_arguments
from script.asr.decode import GreedyDecoder, calculate_cer
#from script.models.CRDNN import Apply_CRDNN
from script.models.CRDNN import Encoder_Decoer
from script.models.check_model import model_init, save_model, reload_model
from script.utils.optimizers import adam_optimizer, sgd_optimizer, scheduler
from script.utils.logger import init_logging
from script.nnets.losses import nll_loss, transducer_loss

#matplotlib.use('Qt5Agg')
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

train_cer = []
train_loss = []
dev_loss = []
dev_cer = []
lr_list = []
flag_plot = True


def calculate_loss(ctc_loss, pred, gold, input_lengths, target_lengths):

    print("{}\t{}\t{}\t{}".format(pred.shape, gold.shape, input_lengths.shape, target_lengths.shape))
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
    input_lengths = (input_lengths * pred.shape[1]).int()
    target_lengths = (target_lengths * gold.shape[1]).int()
    log_probs = pred.transpose(0, 1)  # T x B x C
    # print(gold.size())
    targets = gold
    # targets = gold.contiguous().view(-1)  # (B*T)

    """
    log_probs: torch.Size([209, 8, 3793])
    targets: torch.Size([8, 46])
    input_lengths: torch.Size([8])
    target_lengths: torch.Size([8])
    """
    
    # log_probs = F.log_softmax(log_probs, dim=2)
    # log_probs = log_probs.detach().requires_grad_()
    loss = ctc_loss(log_probs.to("cpu"), targets.to("cpu"), input_lengths.to("cpu"), target_lengths.to("cpu"))

    return loss


def train(model, train_load, dev_load, args, start_epoch, dic, logger, device):

    epoch_n = args.epoch_n + 1
    lr_step = (args.lr - 0.0000001)/((epoch_n * len(train_load)) // args.batch_size)
   
    ctc_loss = torch.nn.CTCLoss(blank=0, reduction='mean') 
    optimizer = adam_optimizer(model, args.lr)
    #optimizer = sgd_optimizer(model, args.lr)
    scheduler_lr = scheduler(optimizer, patience=0, cooldown=0)
    #reload_model(os.path.join(args.result_dir, str(args.seed), "save"), model, optimizer)
    
 
    for epoch in range(start_epoch, epoch_n):

        train_load.batch_data()
        model.train()
        logger.info("Epoch {}/{}".format(epoch, epoch_n))
        logger.info("-" * 10)

        print("status: train\t train_load_size:{}".format(len(train_load)))
        for idx in range(len(train_load)):
            data = train_load[idx]         
            # feats, targets, targets_bos, targets_eos, wav_lengths, target_lens, target_bos_lens, target_eos_lens
            feats, targets, targets_bos, targets_eos, wav_lengths, target_lens, target_bos_lens, target_eos_lens = [item.to(device) for item in data]
            print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(feats.shape, targets.shape, targets_bos.shape, targets_eos.shape, wav_lengths.shape, target_lens.shape, target_bos_lens.shape, target_eos_lens.shape))
            output_en, output_de, output_join = model(feats, targets_bos)
            output_en = torch.nn.functional.log_softmax(output_en, dim=len(output_en.shape)-1)
            output_de = torch.nn.functional.log_softmax(output_de, dim=len(output_de.shape)-1)
            # output_join = torch.nn.functional.log_softmax(output_join, dim=len(output_join.shape)-1)
            closs = calculate_loss(ctc_loss, output_en, targets, wav_lengths, target_lens)
            ce_loss = nll_loss(output_de, targets_eos, target_eos_lens)
            # join_loss = transducer_loss(output_join, targets, wav_lengths, target_lens, blank_index=0)
            loss = closs*0.5 +ce_loss*0.5
            # loss = closs*0.34 +ce_loss*0.33 + join_loss*0.33
            #1.  先看loss是不是nan,如果loss是nan,那么说明可能是在forward的过程中出现了第一条列举的除0或者log0的操作
            # loss = ctc_loss(output, y_train, wavlist_length, label_length, 0)
            assert torch.isnan(loss).sum != 0, logger.info("1.loss:{}".format(loss))
            optimizer.zero_grad()
            closs.backward()
            # 2. 如果loss不是nan,那么说明forward过程没问题，可能是梯度爆炸，所以用梯度裁剪试试
            #torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=3, norm_type=2)
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters()), max_norm=2)
            # 3 在step之前，判断参数是不是nan, 如果不是判断step之后是不是nan
            assert sum([torch.isnan(layer).sum().item() for name, layer in model.named_parameters()]) ==0, logger.info("3.model.mu")
            optimizer.step()
            # 4 在step之后判断，参数和其梯度是不是nan，如果3不是nan,而4是nan,特别是梯度出现了Nan,考虑学习速率是否太大，调小学习速率或者换个优化试试
            assert  sum([torch.isnan(layer).sum().item() for name, layer in model.named_parameters()]) ==0, logger.info("4.model.mu")
            #for name, layer in model.named_parameters():
            #   logger.info("name:{}\nlayer:{}".format(name, layer))

            targ, pred = GreedyDecoder(output_en, targets, wav_lengths, target_lens, dic)
            cer = calculate_cer(targ, pred)
            loss =loss.detach()/ args.batch_size
            
            logger.info("batchs:{}   Loss:{:.2f}   CER:{:.2f}   lr:{:.8f}".format(idx+1, loss.item(), cer, optimizer.state_dict()['param_groups'][0]['lr']))
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr - lr_step * (idx + 1 + (epoch-1) * len(train_load))

        with torch.no_grad():
            dloss, dcer = evaluate(model, dev_load, ctc_loss, args, dic, device)

            # scheduler_lr.step(dcer)
            #if min_loss > dloss:
            save_model(model.encoder, optimizer, epoch, os.path.join(args.result_dir, str(args.seed), 'save'))
            #    min_loss = dloss
            logger.info("Loss:{:.2f}\tDEV CER is:{:.2f}\tlr:{}".format(dloss, dcer, optimizer.state_dict()['param_groups'][0]['lr']))


def evaluate(model, eval_data, ctc_loss, args, dic, device):
    eval_data.batch_data()
    model.eval()
    dloss = 0
    dcer = 0

    for idx in range(len(eval_data)):
        data = eval_data[idx]
        feats, targets, targets_bos, targets_eos, wav_lengths, target_lens, target_bos_lens, target_eos_lens = [item.to(device) for item in data]
        print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(feats.shape, targets.shape, targets_bos.shape, targets_eos.shape, wav_lengths.shape, target_lens.shape, target_bos_lens.shape, target_eos_lens.shape))
        output_en, output_de, output_join = model(feats, targets_bos)
        output_en = torch.nn.functional.log_softmax(output_en, dim=len(output_en.shape)-1)
        output_de = torch.nn.functional.log_softmax(output_de, dim=len(output_de.shape)-1)
        # output_join = torch.nn.functional.log_softmax(output_join, dim=len(output_join.shape)-1)
        closs = calculate_loss(ctc_loss, output_en, targets, wav_lengths, target_lens)
        ce_loss = nll_loss(output_de, targets_eos, target_eos_lens)
        # join_loss = transducer_loss(output_join, targets, wav_lengths, target_lens, blank_index=0)
        # loss = closs*0.34 +ce_loss*0.33 + join_loss*0.33
        loss = closs*0.5 +ce_loss*0.5
        targ, pred = GreedyDecoder(output_en, targets, wav_lengths, target_lens, dic) 
        cer = calculate_cer(targ, pred)
        loss = loss.detach() /args.batch_size
        dloss += loss.item()
        dcer += cer

    dev_cer.append(dcer / len(eval_data))
    dev_loss.append(dloss / len(eval_data))

    return dloss / len(eval_data), dcer / len(eval_data)


def visualization_of_deep_learning_training(start_epoch, epoch_n):
    xx = [i for i in range(start_epoch, epoch_n)]

    fig = plt.figure()

    trloss = fig.add_subplot(231)
    trloss.set(xlim=[0, epoch_n + 20],
               ylim=[np.min(train_loss) - 5 if np.min(train_loss) < 0. else 0, np.max(train_loss) + 20],
               title="train_loss",
               ylabel="loss", xlabel='epoch')
    trloss.plot(xx, train_loss, color='darkred')

    devloss = fig.add_subplot(232)
    devloss.set(xlim=[0, epoch_n + 20],
                ylim=[np.min(dev_loss) - 5 if np.min(dev_loss) < 0. else 0, np.max(dev_loss) + 20],
                title="dev_loss",
                ylabel="loss", xlabel='epoch')
    devloss.plot(xx, dev_loss, color='g')

    trcer = fig.add_subplot(233)
    trcer.set(xlim=[0, epoch_n + 20], ylim=[0, np.max(train_cer) + 5 if np.max(train_cer) > 100 else 100],
              title="train_cer",
              ylabel="cer", xlabel='epoch')
    trcer.plot(xx, train_cer, color='red')

    dcer = fig.add_subplot(234)
    dcer.set(xlim=[0, epoch_n + 20], ylim=[0, np.max(dev_cer) + 5 if np.max(dev_cer) > 100 else 100], title="dev_cer",
             ylabel="cer", xlabel='epoch')
    dcer.plot(xx, dev_cer, color='gray')

    plt.show()

def load_data(args, phones):

    dirpath = os.path.join(args.result_dir, str(args.seed))
    #phones = read_phones(os.path.join(dirpath, 'phones.txt'))

    my_data = My_Data(os.path.join(dirpath, "train.json"), phones, max_during=args.max_during, max_trans_len=args.max_trans_len)
    train_data = DataLoader(dataset=my_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    my_data = My_Data(os.path.join(dirpath, "test.json"), phones)
    test_data = DataLoader(dataset=my_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=True)

    my_data = My_Data(os.path.join(dirpath, "dev.json"), phones)
    dev_data = DataLoader(dataset=my_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    return train_data, test_data, dev_data


def main():
    args = parse_arguments()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    #prepare data
    dirpath = os.path.join(args.result_dir, str(args.seed))
    prepare_data(args.data_folder, dirpath)
    dic = create_phones(dirpath)

    logger = init_logging("train", dirpath)

    if args.feat_type == 'mfcc':
        input_dim = args.feat_cof * 3
    elif args.feat_type == 'fbank':
        input_dim = args.feat_cof

    output_dim = len(dic)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device = torch.device(args.cuda_device)
    else:
        device = torch.device('cpu')

    model = Encoder_Decoer(output_dim, feat_shape=[args.batch_size, args.max_during*100, input_dim])
    model.to(device)
    
    logger.info("\nresult_path:{}\nfeat_type:{}\nfeat_cof:{}\ndevice:{}\nbatch_size:{}\nclassify_num:{}\n"
    .format(dirpath, args.feat_type, args.feat_cof, device, args.batch_size, input_dim))

    logger.info(model)

    model_init(model, init_method="kaiming")
    start_epoch = 1
    train_data, test_data, dev_data = load_data_v2(args, dic)    
    
    logger.info("\ntrain_num:{}\ndev_num:{}\ntest_num:{}\n".format(len(train_data), len(dev_data), len(test_data)))
    train(model, train_data, dev_data, args, start_epoch, dic, logger, device)
    visualization_of_deep_learning_training(start_epoch, args.epoch_n)


if __name__ == "__main__":
    main()
