import numpy as np
import torch

def mixup_data(batch_x, batch_y, alpha=0.5, use_cuda=True):
    '''
    batch_x：batch samples，shape=[batch_size,channels,width,height]
    batch_y：batch labels，shape=[batch_size]
    alpha：parameters of beta distribution
    use_cuda：if use cuda?
    returns：
    	mixed inputs, pairs of targets, and lam
    '''

    if alpha > 0:

        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = batch_x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size) 

    mixed_batchx = lam * batch_x + (1 - lam) * batch_x[index, :]



    batch_ya, batch_yb = batch_y, batch_y[index]
    return mixed_batchx, batch_ya, batch_yb, lam


def mixup_criterion(batch_ya, batch_yb, lam):
    return lambda criterion, pred: lam * criterion(pred, batch_ya) + (1 - lam) * criterion(pred, batch_yb)
