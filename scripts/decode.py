import os
import torch
import numpy as np
import torch.nn.functional as F
#args = parse_arguments()
#dic = dicts(args.train_path, args.test_path, args.dev_path)

'''
lists = [25,20,156,156,...,22,21]
dict ={' ':0,'a_1':2,...}
'''
def int2word(lists, dic):
    words = []
    for item in lists:
        for it in list(dic.keys()):
            if item == dic[it]:
               words.append(it)
    return words

'''
text:ai_1 b c a_2
dict ={' ':0,'a_1':2,...}
'''
def word2int(text, dic):
    ints = []
    for item in text:
        ints.append(dic[item])
    return ints


def GreedyDecoder(output, labels, wavlist_length, label_lengths, dic):

    output = F.log_softmax(output, dim=2)
    output = output.data.cpu().numpy()
    pred = np.vstack([output[i, :j] for i, j in enumerate(wavlist_length)])
    arg_maxes = np.argmax(pred, axis=1)
    decodes = []
    targets = []
    dstep = 0
    tstep = 0

    for i in range(wavlist_length.size(0)):
         prev = 0
         pre = []

         targets.append(int2word(labels[tstep:(tstep+label_lengths[i])].tolist(), dic))
         detmp = arg_maxes[dstep:(dstep+wavlist_length[i])]
         tstep += label_lengths[i]
         dstep += wavlist_length[i]
         for i in detmp:
             if i != 0 and i != prev:
                 pre.append(i)
             prev = i

         decodes.append(int2word(pre,dic))
    return decodes, targets
 
