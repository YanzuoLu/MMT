"""
@author: Yanzuo Lu
@email:  luyz5@mail2.sysu.edu.cn
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd

from mmt.distance import cosine_dist, euclidean_dist


def triplet_loss(feats, pids, world_size, dist_metric):
    if world_size > 1:
        feats = SyncFunction.apply(feats)

        output = [torch.zeros_like(pids) for _ in range(world_size)]
        dist.all_gather(output, pids)
        pids = torch.cat(output, dim=0)

    if dist_metric == 'cosine':
        dist_mat = cosine_dist(feats, feats)
    else:
        dist_mat = euclidean_dist(feats, feats)

    N = dist_mat.shape[0]
    is_pos = pids.unsqueeze(-1).expand(N, N).eq(pids.unsqueeze(-1).expand(N, N).t()).float()
    is_neg = pids.unsqueeze(-1).expand(N, N).ne(pids.unsqueeze(-1).expand(N, N).t()).float()

    dist_ap, _ = torch.max(dist_mat * is_pos, dim=1)
    dist_an, _ = torch.min(dist_mat * is_neg + is_pos * 1e12, dim=1)
    dist_apn = torch.stack([dist_ap, dist_an], dim=1)

    y = torch.ones_like(dist_an).long()
    loss = F.cross_entropy(dist_apn, y)
    return loss


def soft_triplet_loss(feats, ref, pids, world_size, dist_metric):
    if world_size > 1:
        feats = SyncFunction.apply(feats)

        output = [torch.zeros_like(ref) for _ in range(world_size)]
        dist.all_gather(output, ref)
        ref = torch.cat(output, dim=0)

        output = [torch.zeros_like(pids) for _ in range(world_size)]
        dist.all_gather(output, pids)
        pids = torch.cat(output, dim=0)

    if dist_metric == 'cosine':
        dist_mat = cosine_dist(feats, feats)
        dist_mat_ref = cosine_dist(ref, ref)
    else:
        dist_mat = euclidean_dist(feats, feats)
        dist_mat_ref = euclidean_dist(ref, ref)

    N = dist_mat.shape[0]
    is_pos = pids.unsqueeze(-1).expand(N, N).eq(pids.unsqueeze(-1).expand(N, N).t()).float()
    is_neg = pids.unsqueeze(-1).expand(N, N).ne(pids.unsqueeze(-1).expand(N, N).t()).float()

    dist_ap, dist_ap_indices = torch.max(dist_mat * is_pos, dim=1)
    dist_an, dist_an_indices = torch.min(dist_mat * is_neg + is_pos * 1e12, dim=1)
    dist_apn = torch.stack([dist_ap, dist_an], dim=1)

    dist_ap_ref = torch.gather(dist_mat_ref, dim=1, index=dist_ap_indices.unsqueeze(-1)).squeeze()
    dist_an_ref = torch.gather(dist_mat_ref, dim=1, index=dist_an_indices.unsqueeze(-1)).squeeze()
    dist_apn_ref = torch.stack([dist_ap_ref, dist_an_ref], dim=1)
    dist_apn_ref = F.softmax(dist_apn_ref, dim=1).detach()

    loss = F.cross_entropy(dist_apn, dist_apn_ref)
    return loss


class SyncFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]

        dist.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor,dim=0)

        return gathered_tensor

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        dist.all_reduce(grad_input, op=dist.ReduceOp.SUM, async_op=False)

        idx_from = dist.get_rank() * ctx.batch_size
        idx_to = (dist.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to]
