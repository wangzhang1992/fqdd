import torch
import torch.nn as nn
from typing import List, Union
import torch.optim.lr_scheduler as lr_sch
from torch.optim.lr_scheduler import _LRScheduler
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

class WarmupLR(_LRScheduler):
    """The WarmupLR scheduler

    This scheduler is almost same as NoamLR Scheduler except for following
    difference:

    NoamLR:
        lr = optimizer.lr * model_size ** -0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)
    WarmupLR:
        lr = optimizer.lr * warmup_step ** 0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)

    Note that the maximum lr equals to optimizer.lr in this scheduler.

    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: Union[int, float, List[Union[int, float]]] = 25000,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        # __init__() must be invoked before setting field
        # because step() is also invoked in __init__()
        super().__init__(optimizer, last_epoch)

    def __repr__(self):
        return f"{self.__class__.__name__}(warmup_steps={self.warmup_steps})"

    def get_lr(self):
        step_num = self.last_epoch + 1
        warmup_steps = self.warmup_steps
        if not isinstance(warmup_steps, List):
            warmup_steps = [self.warmup_steps] * len(self.base_lrs)

        def initlr_fn(lr):
            return lr * step_num**-0.5

        def warmuplr_fn(lr, warmup_step):
            return lr * warmup_step**0.5 * min(step_num**-0.5,
                                               step_num * warmup_step**-1.5)

        return [
            initlr_fn(lr) if warmup_steps[i] == 0 else warmuplr_fn(
                lr, warmup_steps[i]) for (i, lr) in enumerate(self.base_lrs)
        ]

    def set_step(self, step: int):
        self.last_epoch = step

