import numpy as np
import torch

def mixup_data(batch_x, batch_y, alpha=0.5, use_cuda=True):
    '''
    batch_x：Number of batch samples, shape=[batch_size,channels,width,height]
    batch_y：Number of batch labels,shape=[batch_size]
    alpha：Generate the beta distribution parameter of lam, and generally take 0.5 for better results.
    use_cuda：if use cuda?
    returns：
    	mixed inputs, pairs of targets, and lam
    '''

    if alpha > 0:
        # alpha=0.5 makes lam have a high probability of taking values near 0 or 1
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = batch_x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)  # Generate a scrambled batch_size index

        # Get the mixed_batchx data, which can be mixed with the same type (the same image) or mixed with different types (different images)
    mixed_batchx = lam * batch_x + (1 - lam) * batch_x[index, :]


    batch_ya, batch_yb = batch_y, batch_y[index]
    return mixed_batchx, batch_ya, batch_yb, lam


def mixup_criterion(batch_ya, batch_yb, lam):
    return lambda criterion, pred: lam * criterion(pred, batch_ya) + (1 - lam) * criterion(pred, batch_yb)
