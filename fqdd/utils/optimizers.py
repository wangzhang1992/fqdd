import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_sch

'''
optimizer
'''
def adam_optimizer(model, lr=0.001):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    return optimizer

def sgd_optimizer(model, lr=0.001):

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    return optimizer

def scheduler(optimizer, patience=0, cooldown=0):
    scheduler_l = lr_sch.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=patience, verbose=False, threshold=0.0001, threshold_mode='rel', min_lr=1e-10, cooldown=cooldown, eps=1e-08)
    return scheduler_l

class warmup_lr(nn.Module):
    def __init__(self, lr_initial, n_warmup_steps: int=20000):
        self.lr_initial = lr_initial
        self.n_warmup_steps = n_warmup_steps
        self.current_lr = lr_initial

        self.n_steps = 0
        self.normalize = 1 / (n_warmup_steps * n_warmup_steps ** -1.5)

    def __call__(self, opt):
        self.n_steps +=1
        current_lr = opt.param_groups[0]["lr"]

        lr = self.lr_initial * self._get_lr_scale()

        # Changing the learning rate within the optimizer
        for param_group in opt.param_groups:
            param_group["lr"] = lr

        self.current_lr = current_lr
        return current_lr, lr
   
    def _get_lr_scale(self):
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return self.normalize * min(
            n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5)
        )
'''
lr_init = 0.01
step = 2500
model = torch.nn.Linear(10,10)
optimizer = adam_optimizer(model, lr_init)
warmup = warmup_lr(lr_init, step)
for i in range(22000):
    lr, lr1 = warmup(optimizer)
    print("lr:{}, lr1:{}".format(lr, lr1))
'''
