from __future__ import absolute_import
from collections import defaultdict
import itertools
import math

import numpy as np
import copy
import random
import torch
from torch.utils.data.sampler import (
    Sampler, SequentialSampler, RandomSampler, SubsetRandomSampler,
    WeightedRandomSampler)


def No_index(a, b):
    assert isinstance(a, list)
    return [i for i, j in enumerate(a) if j != b]


class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, num_instances):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(self.num_samples).tolist()
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            if len(t) >= self.num_instances:
                t = np.random.choice(t, size=self.num_instances, replace=False)
            else:
                t = np.random.choice(t, size=self.num_instances, replace=True)
            ret.extend(t)
        return iter(ret)


class RandomMultipleGallerySampler(Sampler):
    def __init__(self, data_source, num_instances=4):
        self.data_source = data_source
        self.index_pid = defaultdict(int)
        self.pid_cam = defaultdict(list)
        self.pid_index = defaultdict(list)
        self.num_instances = num_instances

        for index, (_, pid, cam) in enumerate(data_source):
            self.index_pid[index] = pid
            self.pid_cam[pid].append(cam)
            self.pid_index[pid].append(index)

        self.pids = list(self.pid_index.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(len(self.pids)).tolist()
        ret = []

        for kid in indices:
            i = random.choice(self.pid_index[self.pids[kid]])

            _, i_pid, i_cam = self.data_source[i]

            # ret.append(i)

            pid_i = self.index_pid[i]
            cams = self.pid_cam[pid_i]
            index = self.pid_index[pid_i]
            select_cams = No_index(cams, i_cam)

            if select_cams:

                if len(select_cams) >= self.num_instances - 1:
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances-1, replace=False)
                else:
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances-1, replace=True)

                ret.append(i)
                for kk in cam_indexes:
                    ret.append(index[kk])

            else:
                select_indexes = No_index(index, i)
                if (not select_indexes): continue
                if len(select_indexes) >= self.num_instances - 1:
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=False)
                else:
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=True)

                ret.append(i)
                for kk in ind_indexes:
                    ret.append(index[kk])


        return iter(ret)


class RandomCamAwareSampler(Sampler):
    def __init__(self, dataset, mini_batch_size=64, num_instances=4, rank=0, world_size=1, seed=0):
        self.dataset = dataset
        self.num_instances = num_instances
        self.rank = rank
        self.world_size = world_size
        self.seed = seed

        self.epoch = -1
        self.batch_size = mini_batch_size * world_size
        self.num_pids_per_batch = self.batch_size // num_instances

        self.id2cam = [None for _ in range(len(dataset))]
        self.pid2id = defaultdict(list)
        for index, info in enumerate(dataset):
            _, pid, camid = info
            self.id2cam[index] = camid
            self.pid2id[pid].append(index)
        self.pids = sorted(list(self.pid2id.keys()))

    def __iter__(self):
        start = self.rank
        yield from itertools.islice(self._infinite_indices(), start, None, self.world_size)

    def __len__(self):
        return len(self.dataset) // self.world_size

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _infinite_indices(self):
        rng = np.random.default_rng(self.epoch + self.seed)
        while True:
            pids = rng.permutation(self.pids)
            batch_indices = []

            for pid in pids:
                index = rng.choice(self.pid2id[pid])
                camid = self.id2cam[index]
                diffcam_idxs = [i for i in self.pid2id[pid] if self.id2cam[i] != camid]

                if diffcam_idxs:
                    if len(diffcam_idxs) < self.num_instances - 1:
                        idxs = rng.choice(diffcam_idxs, size=self.num_instances-1, replace=True).tolist()
                    else:
                        idxs = rng.choice(diffcam_idxs, size=self.num_instances-1, replace=False).tolist()
                else:
                    diffid_idxs = [i for i in self.pid2id[pid] if i != index]
                    if diffid_idxs:
                        if len(diffid_idxs) < self.num_instances - 1:
                            idxs = rng.choice(diffid_idxs, size=self.num_instances-1, replace=True).tolist()
                        else:
                            idxs = rng.choice(diffid_idxs, size=self.num_instances-1, replace=False).tolist()
                    else:
                        continue

                batch_indices.append(index)
                batch_indices.extend(idxs)

                if len(batch_indices) == self.batch_size:
                    yield from batch_indices
                    batch_indices = []