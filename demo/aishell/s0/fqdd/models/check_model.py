import torch
import os, sys
import torch.nn as nn
import time


'''
init model
'''
def model_init(model, init_method="default"):
   
    # 在ReLU网络中，假定每一层有一半的神经元被激活，另一半为0。推荐在ReLU网络中使用
    if init_method == "kaiming":
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')

    # 通用方法，适用于任何激活函数
    elif init_method == "default":
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)

    #正交初始化（Orthogonal Initialization）
    # 解决深度网络下的梯度消失、梯度爆炸问题，在RNN中经常使用的参数初始化方法
    elif init_method == "Orthogonal":
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal(m.weight)
    else:
        print("model init method error")
        sys.exit()
'''
save model
'''
def save_model(model, optimizer,  epoch, save_dir):
    
    try:
        #epoch
        os.makedirs(save_dir, exist_ok=True)
        epoch_path = os.path.join(save_dir, 'checkpoint')
        check_epoch = {'checkpoint': epoch}
        torch.save(check_epoch, epoch_path, _use_new_zipfile_serialization=False)

        #model
        model_path = os.path.join(save_dir, 'model.ckpt')
        check_model = {'model': model.state_dict()}
        torch.save(check_model, model_path, _use_new_zipfile_serialization=False)

        #optimizer
        optimizer_path = os.path.join(save_dir, 'optimizer.ckpt')
        check_optimizer = {'optimizer': optimizer.state_dict()}
        torch.save(check_optimizer, optimizer_path, _use_new_zipfile_serialization=False)
    except:
        print('save {}th epoch mode error'.format(epoch))
        return

'''
reload model
'''
def reload_model(load_dir, model=None, optimizer=None, map_location=None):

    print("ready to load pretrain model")

    # load epoch
    if not os.path.exists(os.path.join(load_dir, 'checkpoint')):
       return 0

    try:
       epoch_path = os.path.join(load_dir, 'checkpoint')
       start_epoch = torch.load(epoch_path, map_location=map_location)['checkpoint']
    except:
       print('reload_checkpoint error:\t{}'.format(epoch_path))
       return 0

    # load net params
    try:
       if model:
           model_path = os.path.join(load_dir, 'model.ckpt')
           check_model= torch.load(model_path, map_location=map_location)
           model.load_state_dict(check_model['model'])
    except:
       print('reload_model error:\t{}'.format(model_path))

    # load optimizer
    try:
       if optimizer:
           optimizer_path = os.path.join(load_dir, 'optimizer.ckpt')
           check_optimizer = torch.load(optimizer_path, map_location=map_location)
           optimizer.load_state_dict(check_optimizer['optimizer'])
    except:
       print('reload_optimizer error:\t{}'.format(optimizer_path))
    return start_epoch

def infer_model(load_dir, model):
    # load epoch
    try:
       model_path = os.path.join(load_dir, 'model.ckpt')
       check_model= torch.load(model_path)
       model.load_state_dict(check_model['model'])
    except:
       print('reload_model, file not exists:\t{}'.format(epoch_path))
