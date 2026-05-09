import math
from utils import logger
import numpy as np

class CosineScheduler(object):
    def __init__(self,
                warmup_iterations=500,
                max_iterations=20000,
                min_lr = 1e-5,
                max_lr = 1e-4,
                warmup_init_lr = 1e-7,
                period_epochs = 50,
                lr_multipliers = None,
                is_iter_based = False
                ):
        super(CosineScheduler).__init__()
        self.round_places = 8
        self.lr_multipliers = lr_multipliers
        self.min_lr = min_lr
        self.max_lr = max_lr
        #*
        self.warmup_iterations = max(warmup_iterations, 0)
        if self.warmup_iterations > 0:
            self.warmup_init_lr = warmup_init_lr
            self.warmup_step = (self.max_lr - self.warmup_init_lr) / self.warmup_iterations
        #*
        self.period = (max_iterations - self.warmup_iterations + 1) if is_iter_based \
            else period_epochs
        self.is_iter_based = is_iter_based

    def update_lr(self, optimizer, epoch: int, curr_iter: int,task_name=None):
        lr = self.get_lr(epoch=epoch, curr_iter=curr_iter)
        lr = max(0.0, lr)
        if self.lr_multipliers is not None:
            #assert len(self.lr_multipliers) == len(optimizer.param_groups)
            for g_id, param_group in enumerate(optimizer.param_groups):
                param_group['lr'] = round(lr * self.lr_multipliers[g_id], self.round_places)
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = round(lr, self.round_places)

        return optimizer

    def get_lr(self, epoch: int, curr_iter: int) -> float:
        if curr_iter < self.warmup_iterations:
            curr_lr = self.warmup_init_lr + curr_iter * self.warmup_step
        else:
            if self.is_iter_based:
                curr_iter = curr_iter - self.warmup_iterations
                curr_lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
                            1 + math.cos(math.pi * curr_iter / self.period))
            else:
                curr_lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
                        1 + math.cos(math.pi * epoch / self.period))
        return max(0.0, curr_lr)
    @staticmethod
    def retrieve_lr(optimizer) -> list:
        lr_list = []
        for param_group in optimizer.param_groups:
            lr_list.append(param_group['lr'])
        return lr_list

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    scheduler = CosineScheduler()
    plt.figure()
    max_epoch = 200
    iters = 3785
    cur_lr_list = []
    iter = 0
    for epoch in range(max_epoch):
        print('epoch_{}'.format(epoch))
        for batch in range(iters):

            cur_lr = scheduler.get_lr(epoch, iter)
            iter = iter + 1
            print('cur_lr:', cur_lr)
        cur_lr_list.append(cur_lr)
        print('epoch_{}_end'.format(epoch))
    x_list = list(range(len(cur_lr_list)))
    plt.plot(x_list, cur_lr_list)
    plt.show()
    t = 5