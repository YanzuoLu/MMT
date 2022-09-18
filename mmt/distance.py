"""
@author: Yanzuo Lu
@email:  luyz5@mail2.sysu.edu.cn
"""

import torch
import torch.nn.functional as F


def cosine_dist(x, y):
    x = F.normalize(x)
    y = F.normalize(y)
    dist = 1. - torch.mm(x, y.t())
    dist = dist.clamp(min=1e-12)
    return dist


def euclidean_dist(x, y):
    m, n = x.shape[0], y.shape[0]
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy - 2. * torch.mm(x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()
    return dist