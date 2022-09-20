from __future__ import print_function, absolute_import
import time

import torch
import torch.nn as nn
from torch.nn import functional as F

from .evaluation_metrics import accuracy
from .loss import TripletLoss, CrossEntropyLabelSmooth, SoftTripletLoss, SoftEntropy
from .utils.meters import AverageMeter
from mmt.triplet_loss import triplet_loss, soft_triplet_loss
import torch.distributed as dist

class PreTrainer(object):
    def __init__(self, model, num_classes, margin=0.0):
        super(PreTrainer, self).__init__()
        self.model = model
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes).cuda()
        self.criterion_triple = SoftTripletLoss(margin=margin).cuda()

    def train(self, epoch, data_loader_source, data_loader_target, optimizer, train_iters=200, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_ce = AverageMeter()
        losses_tr = AverageMeter()
        # precisions = AverageMeter()

        end = time.time()

        for i in range(train_iters):
            source_inputs = data_loader_source.next()
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            s_inputs, targets = self._parse_data(source_inputs)
            t_inputs, _ = self._parse_data(target_inputs)
            s_features, s_cls_out = self.model(s_inputs)
            # target samples: only forward
            _ = self.model(t_inputs)

            # backward main #
            # loss_ce, loss_tr, prec1 = self._forward(s_features, s_cls_out, targets)
            loss_ce = F.cross_entropy(s_cls_out, targets, label_smoothing=0.1)
            loss_tr = triplet_loss(s_features, targets, 1, 'euclidean')
            loss = loss_ce + loss_tr

            losses_ce.update(loss_ce.item())
            losses_tr.update(loss_tr.item())
            # precisions.update(prec1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if ((i + 1) % print_freq == 0):
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} ({:.3f})\t'
                      'Loss_tr {:.3f} ({:.3f})\t'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_tr.val, losses_tr.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets

    def _forward(self, s_features, s_outputs, targets):
        loss_ce = self.criterion_ce(s_outputs, targets)
        if isinstance(self.criterion_triple, SoftTripletLoss):
            loss_tr = self.criterion_triple(s_features, s_features, targets)
        elif isinstance(self.criterion_triple, TripletLoss):
            loss_tr, _ = self.criterion_triple(s_features, targets)
        prec, = accuracy(s_outputs.data, targets.data)
        prec = prec[0]

        return loss_ce, loss_tr, prec

class ClusterBaseTrainer(object):
    def __init__(self, model, num_cluster=500):
        super(ClusterBaseTrainer, self).__init__()
        self.model = model
        self.num_cluster = num_cluster

        self.criterion_ce = CrossEntropyLabelSmooth(num_cluster).cuda()
        self.criterion_tri = SoftTripletLoss(margin=0.0).cuda()

    def train(self, epoch, data_loader_target, optimizer, print_freq=1, train_iters=200):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses_ce = AverageMeter()
        losses_tri = AverageMeter()
        precisions = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs, targets = self._parse_data(target_inputs)

            # forward
            f_out_t, p_out_t = self.model(inputs)
            p_out_t = p_out_t[:,:self.num_cluster]

            loss_ce = self.criterion_ce(p_out_t, targets)
            loss_tri = self.criterion_tri(f_out_t, f_out_t, targets)
            loss = loss_ce + loss_tri

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            prec, = accuracy(p_out_t.data, targets.data)

            losses_ce.update(loss_ce.item())
            losses_tri.update(loss_tri.item())
            precisions.update(prec[0])

            # print log #
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} ({:.3f})\t'
                      'Loss_tri {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(data_loader_target),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_tri.val, losses_tri.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets


class MMTTrainer(object):
    # def __init__(self, model_1, model_2,
    #                    model_1_ema, model_2_ema, num_cluster=500, alpha=0.999):
    def __init__(self, model, num_cluster=1000, alpha=0.999):
        super(MMTTrainer, self).__init__()
        # self.model_1 = model_1
        # self.model_2 = model_2
        self.model = model
        self.num_cluster = num_cluster

        # self.model_1_ema = model_1_ema
        # self.model_2_ema = model_2_ema
        self.alpha = alpha

        # self.criterion_ce = CrossEntropyLabelSmooth(num_cluster).cuda()
        # self.criterion_ce_soft = SoftEntropy().cuda()
        # self.criterion_tri = SoftTripletLoss(margin=0.0).cuda()
        # self.criterion_tri_soft = SoftTripletLoss(margin=None).cuda()

    def train(self, epoch, data_loader_target,
            optimizer, world_size, ce_soft_weight=0.5, tri_soft_weight=0.5, print_freq=1, train_iters=200):
        # self.model_1.train()
        # self.model_2.train()
        # self.model_1_ema.train()
        # self.model_2_ema.train()
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses_ce = [AverageMeter(),AverageMeter()]
        losses_tri = [AverageMeter(),AverageMeter()]
        losses_ce_soft = AverageMeter()
        losses_tri_soft = AverageMeter()
        # precisions = [AverageMeter(),AverageMeter()]

        data_loader_target_iter = iter(data_loader_target)

        end = time.time()
        for i in range(train_iters):
            # target_inputs = data_loader_target.next()
            # print(f'before data next {dist.get_rank()}')
            target_inputs = next(data_loader_target_iter)

            # process inputs
            # print(f'before move to gpu {dist.get_rank()}')
            inputs_1, inputs_2, targets = self._parse_data(target_inputs)

            data_time.update(time.time() - end)

            # print(f'before forward {dist.get_rank()}')
            loss, loss_ce_1, loss_ce_2, loss_tri_1, loss_tri_2, loss_ce_soft, loss_tri_soft = \
                self.model(inputs_1, inputs_2, targets, world_size)

            # feats_1 = self.avg_pool(self.backbone_1(imgs_1))
            # logits_1 = self.classifier_1(self.bn_neck_1(feats_1))[:,:self.num_clusters]
            # feats_2 = self.avg_pool(self.backbone_2(imgs_2))
            # logits_2 = self.classifier_2(self.bn_neck_2(feats_2))[:,:self.num_clusters]
            # f_out_t1, p_out_t1 = self.model_1(inputs_1)
            # f_out_t2, p_out_t2 = self.model_2(inputs_2)
            # p_out_t1 = p_out_t1[:,:self.num_cluster]
            # p_out_t2 = p_out_t2[:,:self.num_cluster]

            # with torch.no_grad():
                # feats_1_ema = self.avg_pool(self.backbone_1_ema(imgs_1))
                # logits_1_ema = self.classifier_1_ema(self.bn_neck_1_ema(feats_1_ema))[:,:self.num_clusters]
                # feats_2_ema = self.avg_pool(self.backbone_2_ema(imgs_2))
                # logits_2_ema = self.classifier_2_ema(self.bn_neck_2_ema(feats_2_ema))[:,:self.num_clusters]
                # f_out_t1_ema, p_out_t1_ema = self.model_1_ema(inputs_1)
                # f_out_t2_ema, p_out_t2_ema = self.model_2_ema(inputs_2)
                # p_out_t1_ema = p_out_t1_ema[:,:self.num_cluster]
                # p_out_t2_ema = p_out_t2_ema[:,:self.num_cluster]

            # id_loss = F.cross_entropy(logits_1, pids, label_smoothing=self.label_smoothing) + \
            #     F.cross_entropy(logits_2, pids, label_smoothing=self.label_smoothing)
            # loss_ce_1 = F.cross_entropy(p_out_t1, targets, label_smoothing=0.1)
            # loss_ce_2 = F.cross_entropy(p_out_t2, targets, label_smoothing=0.1)

            # tri_loss = triplet_loss(feats_1, pids, self.world_size, self.dist_metric) + \
            #     triplet_loss(feats_2, pids, self.world_size, self.dist_metric)
            # loss_tri_1 = triplet_loss(f_out_t1, targets, world_size, 'euclidean')
            # loss_tri_2 = triplet_loss(f_out_t2, targets, world_size, 'euclidean')

            # soft_id_loss = F.cross_entropy(logits_1, F.softmax(logits_2_ema, dim=1).detach()) + \
            #     F.cross_entropy(logits_2, F.softmax(logits_1_ema, dim=1).detach())
            # soft_tri_loss = soft_triplet_loss(feats_1, feats_2_ema, pids, self.world_size, self.dist_metric) + \
            #     soft_triplet_loss(feats_2, feats_1_ema, pids, self.world_size, self.dist_metric)
            # loss_ce_soft = F.cross_entropy(p_out_t1, F.softmax(p_out_t2_ema, dim=1).detach()) + \
            #     F.cross_entropy(p_out_t2, F.softmax(p_out_t1_ema, dim=1).detach())
            # loss_tri_soft = soft_triplet_loss(f_out_t1, f_out_t2_ema, targets, world_size, 'euclidean') + \
            #     soft_triplet_loss(f_out_t2, f_out_t1_ema, targets, world_size, 'euclidean')

            # loss = self.soft_id_loss_weight * soft_id_loss + (1. - self.soft_id_loss_weight) * id_loss + \
            #     self.soft_tri_loss_weight * soft_tri_loss + (1. - self.soft_tri_loss_weight) * tri_loss

            # -------------------------------------------------------------------------------------------------------------
            # # forward
            # f_out_t1, p_out_t1 = self.model_1(inputs_1)
            # f_out_t2, p_out_t2 = self.model_2(inputs_2)
            # p_out_t1 = p_out_t1[:,:self.num_cluster]
            # p_out_t2 = p_out_t2[:,:self.num_cluster]

            # f_out_t1_ema, p_out_t1_ema = self.model_1_ema(inputs_1)
            # f_out_t2_ema, p_out_t2_ema = self.model_2_ema(inputs_2)
            # p_out_t1_ema = p_out_t1_ema[:,:self.num_cluster]
            # p_out_t2_ema = p_out_t2_ema[:,:self.num_cluster]

            # loss_ce_1 = self.criterion_ce(p_out_t1, targets)
            # loss_ce_2 = self.criterion_ce(p_out_t2, targets)

            # loss_tri_1 = self.criterion_tri(f_out_t1, f_out_t1, targets)
            # loss_tri_2 = self.criterion_tri(f_out_t2, f_out_t2, targets)

            # loss_ce_soft = self.criterion_ce_soft(p_out_t1, p_out_t2_ema) + self.criterion_ce_soft(p_out_t2, p_out_t1_ema)
            # loss_tri_soft = self.criterion_tri_soft(f_out_t1, f_out_t2_ema, targets) + \
            #                 self.criterion_tri_soft(f_out_t2, f_out_t1_ema, targets)

            # loss = (loss_ce_1 + loss_ce_2)*(1-ce_soft_weight) + \
            #          (loss_tri_1 + loss_tri_2)*(1-tri_soft_weight) + \
            #          loss_ce_soft*ce_soft_weight + loss_tri_soft*tri_soft_weight

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # self._update_ema_variables(self.model_1, self.model_1_ema, self.alpha, epoch*len(data_loader_target)+i)
            # self._update_ema_variables(self.model_2, self.model_2_ema, self.alpha, epoch*len(data_loader_target)+i)
            # self._update_ema_variables(self.model_1, self.model_1_ema, self.alpha, epoch*train_iters+i)
            # self._update_ema_variables(self.model_2, self.model_2_ema, self.alpha, epoch*train_iters+i)
            self.model.module.ema_update(epoch*train_iters+i)

            # prec_1, = accuracy(p_out_t1.data, targets.data)
            # prec_2, = accuracy(p_out_t2.data, targets.data)

            losses_ce[0].update(loss_ce_1.item())
            losses_ce[1].update(loss_ce_2.item())
            losses_tri[0].update(loss_tri_1.item())
            losses_tri[1].update(loss_tri_2.item())
            losses_ce_soft.update(loss_ce_soft.item())
            losses_tri_soft.update(loss_tri_soft.item())
            # precisions[0].update(prec_1[0])
            # precisions[1].update(prec_2[0])

            # print log #
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} / {:.3f}\t'
                      'Loss_tri {:.3f} / {:.3f}\t'
                      'Loss_ce_soft {:.3f}\t'
                      'Loss_tri_soft {:.3f}\t'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce[0].avg, losses_ce[1].avg,
                              losses_tri[0].avg, losses_tri[1].avg,
                              losses_ce_soft.avg, losses_tri_soft.avg))

    def _update_ema_variables(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        with torch.no_grad():
            for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                ema_param.mul_(alpha).add_(param, alpha=1.-alpha)

    def _parse_data(self, inputs):
        imgs_1, imgs_2, pids = inputs
        inputs_1 = imgs_1.cuda(dist.get_rank(), non_blocking=True)
        inputs_2 = imgs_2.cuda(dist.get_rank(), non_blocking=True)
        targets = pids.cuda(dist.get_rank(), non_blocking=True)
        return inputs_1, inputs_2, targets
