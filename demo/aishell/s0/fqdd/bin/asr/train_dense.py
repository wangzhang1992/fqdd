import torch
import os,sys
import numpy as np
sys.path.insert(0, "./")
import torch.optim.lr_scheduler as lr_sch
import matplotlib.pyplot as plt
import matplotlib
# from apex import amp
from tqdm import tqdm
from script.utils.feature import get_feats
from script.prepare_data.aishell_prepare_data import prepare_data
from script.utils.lang import create_phones, read_phones
from script.utils.load_data import load_data_v2
from script.utils.argument import parse_arguments
from script.asr.decode import GreedyDecoder, calculate_cer
#from script.models.CRDNN import Apply_CRDNN
from script.models.CRDNN import Encoder_Decoer
from script.models.check_model import model_init, save_model, reload_model
from script.utils.optimizers import adam_optimizer, sgd_optimizer, scheduler, warmup_lr
from script.utils.logger import init_logging
from script.nnets.losses import nll_loss, transducer_loss
from script.models.dense import densenet_custom
from script.utils.edit_distance import Static_Cer
flag_plot = True


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

    min_loss = 1000
    lr_cer = 100
    slice_len = 5
    slice_step = slice_len * args.batch_size
    epoch_n = args.epoch_n
    ctc_loss = torch.nn.CTCLoss(blank=0, reduction='mean') 
    optimizer = adam_optimizer(model, args.lr)
    #optimizer = sgd_optimizer(model, args.lr)
    # scheduler_lr = scheduler(optimizer, patience=0, cooldown=0)
    warm_up = warmup_lr(args.lr, 2500)
    start_epoch = reload_model(os.path.join(args.result_dir, str(args.seed), "save"), model, optimizer)
    
 
    for epoch in range(start_epoch, epoch_n+1):

        train_load.batch_data()
        model.train()
        logger.info("Epoch {}/{}".format(epoch, epoch_n))
        logger.info("-" * 10)

        stat_cer = Static_Cer()
        stat_loss = 0
        print("status: train\t train_load_size:{}".format(len(train_load)))
        for idx in tqdm(range(len(train_load))):
            data = train_load[idx]         
            # feats, targets, targets_bos, targets_eos, wav_lengths, target_lens, target_bos_lens, target_eos_lens
            feats, targets, targets_bos, targets_eos, wav_lengths, target_lens, target_bos_lens, target_eos_lens = [item.to(device) for item in data]
            #print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(feats.shape, targets.shape, targets_bos.shape, targets_eos.shape, wav_lengths.shape, target_lens.shape, target_bos_lens.shape, target_eos_lens.shape))
            output = model(feats)
            #output = torch.nn.functional.log_softmax(output, dim=-1)
            loss = calculate_loss(ctc_loss, output, targets, wav_lengths, target_lens)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters()), 5)
            optimizer.step()
            warm_up(optimizer)
            targ, pred = GreedyDecoder(output, targets, wav_lengths, target_lens, dic)
            
            cer = stat_cer.batch_cer(targ, pred)
            loss = loss.detach()
            stat_loss += loss.item()
            loss = loss.item()
            
            logger.info("batchs:{}   Loss:{:.2f}   CER:{:.2f}".format(idx+1, loss, cer))
        cer, corr, det, ins, sub = stat_cer.static_cer()
        loss = stat_loss / len(train_load)
        # scheduler_lr.step(loss)
        save_model(model, optimizer, epoch, os.path.join(args.result_dir, str(args.seed), 'save'))
        logger.info("loss:{}, cer:{}, lr:{}, corr:{}, det:{}, ins:{}, sub:{}".format(loss, cer, optimizer.state_dict()['param_groups'][0]['lr'], corr, det, ins, sub))
        '''
        with torch.no_grad():
            loss, cer, corr, det, ins, sub = evaluate(model, dev_load, ctc_loss, args, dic, device)

            #scheduler_lr.step(loss)
            save_model(model, optimizer, epoch, os.path.join(args.result_dir, str(args.seed), 'save'))
            logger.info("DEV:  loss:{}, cer:{}, lr:{} corr:{}, det:{}, ins:{}, sub:{}".format(loss, cer, optimizer.state_dict()['param_groups'][0]['lr'], corr, det, ins, sub))
        '''
def evaluate(model, eval_data, ctc_loss, args, dic, device):
    eval_data.batch_data()
    model.eval()
    dloss = 0
    dev_cer = Static_Cer()

    for idx in tqdm(range(len(eval_data))):
        data = eval_data[idx]
        feats, targets, targets_bos, targets_eos, wav_lengths, target_lens, target_bos_lens, target_eos_lens = [item.to(device) for item in data]
        #print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(feats.shape, targets.shape, targets_bos.shape, targets_eos.shape, wav_lengths.shape, target_lens.shape, target_bos_lens.shape, target_eos_lens.shape))
        output = model(feats)
        #output = torch.nn.functional.log_softmax(output, dim=-1)
        loss = calculate_loss(ctc_loss, output, targets, wav_lengths, target_lens)
        targ, pred = GreedyDecoder(output, targets, wav_lengths, target_lens, dic) 
        cer = dev_cer.batch_cer(targ, pred)
        loss = loss.detach()
        dloss += loss.item()
    loss = dloss / len(eval_data)
    cer, corr, det, ins, sub = dev_cer.static_cer()
    return loss, cer, corr, det, ins, sub


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
        if args.feat_cof==40:
            input_dim = args.feat_cof
        else:
            input_dim = args.feat_cof *3
    elif args.feat_type == 'fbank':
        input_dim = args.feat_cof * 2

    output_dim = len(dic)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device = torch.device(args.cuda_device)
    else:
        device = torch.device('cpu')

    model = densenet_custom(num_classes=output_dim) 
    #model = Encoder_Decoer(output_dim, feat_shape=[args.batch_size, args.max_during*100, input_dim])
    model.to(device)
    
    logger.info("\nresult_path:{}\nfeat_type:{}\ninput_dim:{}\ndevice:{}\nbatch_size:{}\nclass_num:{}\n"
    .format(dirpath, args.feat_type, input_dim, device, args.batch_size, len(dic)))

    logger.info(model)

    model_init(model, init_method="kaiming")
    start_epoch = 1
    train_data, test_data, dev_data = load_data_v2(args, dic)    
    
    logger.info("\ntrain_num:{}\ndev_num:{}\ntest_num:{}\n".format(len(train_data), len(dev_data), len(test_data)))
    train(model, train_data, dev_data, args, start_epoch, dic, logger, device)
    #visualization_of_deep_learning_training(start_epoch, args.epoch_n)


if __name__ == "__main__":
    main()
