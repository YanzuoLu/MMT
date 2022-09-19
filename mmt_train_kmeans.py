import argparse
import builtins
import copy
import datetime
import os
import os.path as osp
import random
import sys
import time
from collections import defaultdict

import numpy as np
from sklearnex import patch_sklearn

patch_sklearn()
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.transforms as transforms
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.transforms.functional import InterpolationMode

from mmt import datasets, models
from mmt.evaluators import Evaluator, extract_features
from mmt.models.resnet50 import resnet50
from mmt.trainers import MMTTrainer
from mmt.utils.data import IterLoader
from mmt.utils.data import transforms as T
from mmt.utils.data.preprocessor import Preprocessor
from mmt.utils.data.sampler import (RandomCamAwareSampler,
                                    RandomMultipleGallerySampler)
from mmt.utils.logging import Logger
from mmt.utils.random_erasing import RandomErasing
from mmt.utils.serialization import (copy_state_dict, load_checkpoint,
                                     save_checkpoint)
from mmt.triplet_loss import triplet_loss, soft_triplet_loss


best_mAP = 0

def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset

def get_train_loader(dataset, height, width, batch_size, workers,
                    num_instances, iters):

    # normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225])
    # train_transformer = T.Compose([
    #          T.Resize((height, width), interpolation=3),
    #          T.RandomHorizontalFlip(p=0.5),
    #          T.Pad(10),
    #          T.RandomCrop((height, width)),
    #          T.ToTensor(),
    #          normalizer,
	#          T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    #      ])

    train_transformer = transforms.Compose([
        transforms.Resize((height, width), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.Pad(10, fill=127),
        transforms.RandomCrop((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02, 0.25))
    ])

    # train_set = sorted(dataset.train)
    train_set = dataset.train
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        # sampler = RandomMultipleGallerySampler(train_set, num_instances)
        sampler = RandomCamAwareSampler(train_set, batch_size // WORLD_SIZE, num_instances, RANK, WORLD_SIZE)
    else:
        sampler = None
    # train_loader = IterLoader(
    #             DataLoader(Preprocessor(train_set, root=dataset.images_dir,
    #                                     transform=train_transformer, mutual=True),
    #                         batch_size=batch_size // WORLD_SIZE, num_workers=workers, sampler=sampler,
    #                         shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)
    train_loader = DataLoader(
        dataset=Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer, mutual=True),
        batch_size=batch_size // WORLD_SIZE,
        num_workers=workers,
        sampler=sampler,
        pin_memory=True
    )

    return train_loader

def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader

def create_model(args):
    # model_1 = models.create(args.arch, num_features=args.features, dropout=args.dropout, num_classes=args.num_clusters)
    # model_2 = models.create(args.arch, num_features=args.features, dropout=args.dropout, num_classes=args.num_clusters)

    # model_1_ema = models.create(args.arch, num_features=args.features, dropout=args.dropout, num_classes=args.num_clusters)
    # model_2_ema = models.create(args.arch, num_features=args.features, dropout=args.dropout, num_classes=args.num_clusters)

    class Model(nn.Module):
        def __init__(self, model_1, model_2, model_1_ema, model_2_ema, num_clusters):
            super().__init__()
            self.model_1 = model_1
            self.model_2 = model_2
            self.model_1_ema = model_1_ema
            self.model_2_ema = model_2_ema
            self.num_clusters = num_clusters

            self.soft_id_loss_weight = 0.5
            self.soft_tri_loss_weight = 0.8
            self.alpha = 0.999

        def forward(self, inputs_1, inputs_2=None, targets=None, world_size=None):
            if not self.training:
                return (self.model_1_ema(inputs_1) + self.model_2_ema(inputs_1)) / 2.

            f_out_t1, p_out_t1 = self.model_1(inputs_1)
            f_out_t2, p_out_t2 = self.model_2(inputs_2)
            p_out_t1 = p_out_t1[:,:self.num_clusters]
            p_out_t2 = p_out_t2[:,:self.num_clusters]

            with torch.no_grad():
                f_out_t1_ema, p_out_t1_ema = self.model_1_ema(inputs_1)
                f_out_t2_ema, p_out_t2_ema = self.model_2_ema(inputs_2)
                p_out_t1_ema = p_out_t1_ema[:,:self.num_clusters]
                p_out_t2_ema = p_out_t2_ema[:,:self.num_clusters]

            loss_ce_1 = F.cross_entropy(p_out_t1, targets, label_smoothing=0.1)
            loss_ce_2 = F.cross_entropy(p_out_t2, targets, label_smoothing=0.1)

            loss_tri_1 = triplet_loss(f_out_t1, targets, world_size, 'euclidean')
            loss_tri_2 = triplet_loss(f_out_t2, targets, world_size, 'euclidean')

            loss_ce_soft = F.cross_entropy(p_out_t1, F.softmax(p_out_t2_ema, dim=1).detach()) + \
                F.cross_entropy(p_out_t2, F.softmax(p_out_t1_ema, dim=1).detach())
            loss_tri_soft = soft_triplet_loss(f_out_t1, f_out_t2_ema, targets, world_size, 'euclidean') + \
                soft_triplet_loss(f_out_t2, f_out_t1_ema, targets, world_size, 'euclidean')

            loss = self.soft_id_loss_weight * loss_ce_soft + (1. - self.soft_id_loss_weight) * (loss_ce_1 + loss_ce_2) + \
                self.soft_tri_loss_weight * loss_tri_soft + (1. - self.soft_tri_loss_weight) * (loss_tri_1 + loss_tri_2)

            return loss, loss_ce_1, loss_ce_2, loss_tri_1, loss_tri_2, loss_ce_soft, loss_tri_soft

        def ema_update(self, steps):
            alpha = min(1 - 1 / (steps + 1), self.alpha)
            self._ema_update(self.model_1, self.model_1_ema, alpha)
            self._ema_update(self.model_2, self.model_2_ema, alpha)

        def _ema_update(self, module_1, module_2, alpha):
            with torch.no_grad():
                for param_1, param_2 in zip(module_1.parameters(), module_2.parameters()):
                    param_2.mul_(alpha).add_(param_1, alpha=1.-alpha)

        def reset_classifier(self, cluster_centers):
            with torch.no_grad():
                self.model_1.classifier.weight[:self.num_clusters].copy_(cluster_centers)
                self.model_2.classifier.weight[:self.num_clusters].copy_(cluster_centers)
                self.model_1_ema.classifier.weight[:self.num_clusters].copy_(cluster_centers)
                self.model_2_ema.classifier.weight[:self.num_clusters].copy_(cluster_centers)

    model_1 = resnet50(pretrained=True, last_stride=1, num_classes=args.num_clusters)
    model_2 = resnet50(pretrained=True, last_stride=1, num_classes=args.num_clusters)
    model_1_ema = copy.deepcopy(model_1)
    model_2_ema = copy.deepcopy(model_2)

    f = torch.load(args.init, map_location='cpu')['model']
    state_dict_1 = {}
    state_dict_2 = {}
    state_dict_1_ema = {}
    state_dict_2_ema = {}
    for n, p in f.items():
        if n.startswith('backbone_1_ema'):
            state_dict_1_ema[n[15:]] = p
        elif n.startswith('backbone_2_ema'):
            state_dict_2_ema[n[15:]] = p
        elif n.startswith('backbone_1'):
            state_dict_1[n[11:]] = p
        elif n.startswith('backbone_2'):
            state_dict_2[n[11:]] = p
        elif n.startswith('bn_neck_1_ema'):
            state_dict_1_ema['bn_neck.' + n[14:]] = p
        elif n.startswith('bn_neck_2_ema'):
            state_dict_2_ema['bn_neck.' + n[14:]] = p
        elif n.startswith('bn_neck_1'):
            state_dict_1['bn_neck.' + n[10:]] = p
        elif n.startswith('bn_neck_2'):
            state_dict_2['bn_neck.' + n[10:]] = p

    print(model_1.load_state_dict(state_dict_1, strict=False))
    print(model_2.load_state_dict(state_dict_2, strict=False))
    print(model_1_ema.load_state_dict(state_dict_1_ema, strict=False))
    print(model_2_ema.load_state_dict(state_dict_2_ema, strict=False))

    model = Model(model_1, model_2, model_1_ema, model_2_ema, args.num_clusters)
    model.cuda(LOCAL_RANK)

    # model_1.cuda()
    # model_2.cuda()
    # model_1_ema.cuda()
    # model_2_ema.cuda()
    # model_1 = nn.DataParallel(model_1)
    # model_2 = nn.DataParallel(model_2)
    # model_1_ema = nn.DataParallel(model_1_ema)
    # model_2_ema = nn.DataParallel(model_2_ema)

    if WORLD_SIZE > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[LOCAL_RANK], find_unused_parameters=True)
        # model_1 = torch.nn.parallel.DistributedDataParallel(model_1, device_ids=[LOCAL_RANK])
        # model_2 = torch.nn.parallel.DistributedDataParallel(model_2, device_ids=[LOCAL_RANK])
        # model_1_ema = torch.nn.parallel.DistributedDataParallel(model_1_ema, device_ids=[LOCAL_RANK])
        # model_2_ema = torch.nn.parallel.DistributedDataParallel(model_2_ema, device_ids=[LOCAL_RANK])

    # initial_weights = load_checkpoint(args.init_1)
    # copy_state_dict(initial_weights['state_dict'], model_1)
    # copy_state_dict(initial_weights['state_dict'], model_1_ema)
    # model_1_ema.module.classifier.weight.data.copy_(model_1.module.classifier.weight.data)

    # initial_weights = load_checkpoint(args.init_2)
    # copy_state_dict(initial_weights['state_dict'], model_2)
    # copy_state_dict(initial_weights['state_dict'], model_2_ema)
    # model_2_ema.module.classifier.weight.data.copy_(model_2.module.classifier.weight.data)

    # return model_1, model_2, model_1_ema, model_2_ema
    return model


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    global best_mAP

    cudnn.benchmark = True
    torch.cuda.set_device(LOCAL_RANK)

    if LOCAL_RANK == 0:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
        print("==========\nArgs:{}\n==========".format(args))

    if WORLD_SIZE > 1:
        dist.init_process_group(backend='nccl')

    # Create data loaders
    iters = args.iters if (args.iters>0) else None
    dataset_target = get_data(args.dataset_target, args.data_dir)
    test_loader_target = get_test_loader(dataset_target, args.height, args.width, args.batch_size, args.workers)
    cluster_loader = get_test_loader(dataset_target, args.height, args.width, args.batch_size, args.workers, testset=dataset_target.train)

    # Create model
    # model_1, model_2, model_1_ema, model_2_ema = create_model(args)
    model = create_model(args)

    # Evaluator
    # evaluator_1_ema = Evaluator(model_1_ema)
    # evaluator_2_ema = Evaluator(model_2_ema)
    evaluator = Evaluator(model.module)

    for epoch in range(args.epochs):
        if RANK == 0:
            # dict_f, _ = extract_features(model_1_ema, cluster_loader, print_freq=50)
            # cf_1 = torch.stack(list(dict_f.values())).numpy()
            # # cf_1 = torch.stack(list(dict_f.values()))
            # dict_f, _ = extract_features(model_2_ema, cluster_loader, print_freq=50)
            # cf_2 = torch.stack(list(dict_f.values())).numpy()
            # # cf_2 = torch.stack(list(dict_f.values()))
            # cf = (cf_1+cf_2)/2
            dict_f = extract_features(model.module, cluster_loader, print_freq=50)
            cf = torch.stack(list(dict_f.values())).numpy()

            start_time = time.time()
            print('\n Clustering into {} classes \n'.format(args.num_clusters))
            # km = KMeans(n_clusters=args.num_clusters, random_state=args.seed).fit(cf)
            km = KMeans(n_clusters=args.num_clusters, random_state=args.seed).fit_predict(cf).tolist()
            print(f'cluster running time {datetime.timedelta(seconds=int(time.time()-start_time))}')

            clusters = defaultdict(list)
            for i, label in enumerate(km):
                if label == -1: continue
                clusters[label].append(torch.from_numpy(cf[i]))
            cluster_centers = [torch.stack(clusters[i]).mean(0) for i in sorted(clusters.keys())]
            cluster_centers = torch.stack(cluster_centers)
            cluster_centers = F.normalize(cluster_centers)
            objects = [cluster_centers, km]
            print('after collecting')
        else:
            objects = [None, None]

        if WORLD_SIZE > 1:
            print(f'before barrier {LOCAL_RANK}')
            dist.barrier()
            print(f'before broadcast {LOCAL_RANK}')
            dist.broadcast_object_list(objects, src=0)

        cluster_centers, km = objects
        print(f'before copy weight {LOCAL_RANK}')
        # model.module.reset_classifier(cluster_centers)
        with torch.no_grad():
            model.module.model_1.classifier.weight[:args.num_clusters].copy_(cluster_centers, non_blocking=True)
            model.module.model_2.classifier.weight[:args.num_clusters].copy_(cluster_centers, non_blocking=True)
            model.module.model_1_ema.classifier.weight[:args.num_clusters].copy_(cluster_centers, non_blocking=True)
            model.module.model_2_ema.classifier.weight[:args.num_clusters].copy_(cluster_centers, non_blocking=True)

        # model_1.module.classifier.weight.data.copy_(torch.from_numpy(normalize(km.cluster_centers_, axis=1)).float().cuda())
        # model_2.module.classifier.weight.data.copy_(torch.from_numpy(normalize(km.cluster_centers_, axis=1)).float().cuda())
        # model_1_ema.module.classifier.weight.data.copy_(torch.from_numpy(normalize(km.cluster_centers_, axis=1)).float().cuda())
        # model_2_ema.module.classifier.weight.data.copy_(torch.from_numpy(normalize(km.cluster_centers_, axis=1)).float().cuda())

        # target_label = km.labels_
        target_label = km

        print('before reconstruct dataset')
        # change pseudo labels
        for i in range(len(dataset_target.train)):
            dataset_target.train[i] = list(dataset_target.train[i])
            dataset_target.train[i][1] = int(target_label[i])
            dataset_target.train[i] = tuple(dataset_target.train[i])

        print('before get train loader')
        train_loader_target = get_train_loader(dataset_target, args.height, args.width,
                                            args.batch_size, args.workers, args.num_instances, iters)
        train_loader_target.sampler.set_epoch(epoch)

        # Optimizer
        if epoch == 0:
            params = [p for p in model.parameters() if p.requires_grad]
            param_groups = [{'params': params, 'lr': args.lr}]
            # for key, value in model_1.named_parameters():
            #     if not value.requires_grad:
            #         continue
            #     params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
            # for key, value in model_2.named_parameters():
            #     if not value.requires_grad:
            #         continue
            #     params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
        optimizer = torch.optim.Adam(param_groups, weight_decay=args.weight_decay)

        # Trainer
        # trainer = MMTTrainer(model_1, model_2, model_1_ema, model_2_ema,
        #                         num_cluster=args.num_clusters, alpha=args.alpha)
        trainer = MMTTrainer(model, num_cluster=args.num_clusters, alpha=args.alpha)

        # train_loader_target.new_epoch()
        print('before train')
        trainer.train(epoch, train_loader_target, optimizer, WORLD_SIZE,
                    ce_soft_weight=args.soft_ce_weight, tri_soft_weight=args.soft_tri_weight,
                    print_freq=args.print_freq, train_iters=iters)

        def save_model(model_ema, is_best, best_mAP, mid):
            save_checkpoint({
                'state_dict': model_ema.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'model'+str(mid)+'_checkpoint.pth.tar'))

        if ((epoch+1)%args.eval_step==0 or (epoch==args.epochs-1)) and RANK == 0:
            # mAP_1 = evaluator_1_ema.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=False)
            # mAP_2 = evaluator_2_ema.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=False)
            # is_best = (mAP_1>best_mAP) or (mAP_2>best_mAP)
            # best_mAP = max(mAP_1, mAP_2, best_mAP)
            # save_model(model_1_ema, (is_best and (mAP_1>mAP_2)), best_mAP, 1)
            # save_model(model_2_ema, (is_best and (mAP_1<=mAP_2)), best_mAP, 2)
            mAP = evaluator.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=False)

            # print('\n * Finished epoch {:3d}  model no.1 mAP: {:5.1%} model no.2 mAP: {:5.1%}  best: {:5.1%}{}\n'.
            #       format(epoch, mAP_1, mAP_2, best_mAP, ' *' if is_best else ''))

        if WORLD_SIZE > 1:
            dist.barrier()

    # if RANK == 0:
    #     print ('Test on the best model.')
    #     checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    #     model_1_ema.load_state_dict(checkpoint['state_dict'])
    #     evaluator_1_ema.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=True)

    # if WORLD_SIZE > 1:
    #     dist.barrier()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MMT Training")
    # data
    parser.add_argument('-dt', '--dataset-target', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--num-clusters', type=int, default=500)
    parser.add_argument('--height', type=int, default=256,
                        help="input height")
    parser.add_argument('--width', type=int, default=128,
                        help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate of new parameters, for pretrained "
                             "parameters it is 10 times smaller than this")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--alpha', type=float, default=0.999)
    parser.add_argument('--moving-avg-momentum', type=float, default=0)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--soft-ce-weight', type=float, default=0.5)
    parser.add_argument('--soft-tri-weight', type=float, default=0.8)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--iters', type=int, default=800)
    # training configs
    parser.add_argument('--init', type=str, default='', metavar='PATH')
    parser.add_argument('--init-1', type=str, default='', metavar='PATH')
    parser.add_argument('--init-2', type=str, default='', metavar='PATH')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--print-freq', type=int, default=1)
    parser.add_argument('--eval-step', type=int, default=1)
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))

    RANK = int(os.environ['RANK']) if torch.cuda.device_count() > 1 else 0
    LOCAL_RANK = int(os.environ['LOCAL_RANK']) if torch.cuda.device_count() > 1 else 0
    WORLD_SIZE = int(os.environ['WORLD_SIZE']) if torch.cuda.device_count() > 1 else 1

    # if LOCAL_RANK == 0:
    #     def print_pass(*args):
    #         pass
    #     builtins.print = print_pass

    main()
