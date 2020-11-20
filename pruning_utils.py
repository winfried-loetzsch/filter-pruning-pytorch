import torch
import numpy as np
from torch import nn


def apply_to_type(m, t, f):
    chd = list(m.children())

    if len(chd) > 0:
        for c in chd:
            apply_to_type(c, t, f)

        return

    if isinstance(m, t):
        f(m)


def remove_at_dim(tensor, idx, dim):
    if dim == 0:
        return torch.cat((tensor[:idx], tensor[idx + 1:]), dim=0)
    elif dim == 1:
        return torch.cat((tensor[:, :idx], tensor[:, idx + 1:]), dim=1)
    else:
        raise ValueError("not implemented")


def select_at_dim(tensor, idxs, dim):
    if dim == 0:
        return tensor[idxs]
    elif dim == 1:
        return tensor[:, idxs]
    else:
        raise ValueError("not implemented")


def prune_bn(bn, idx):
    if not hasattr(bn, "pruning_indices"):
        bn.pruning_indices = []
    bn.pruning_indices.append(idx)


def prune_conv_out(conv, idx):
    if hasattr(conv, "group_pruning_indices"):
        raise RuntimeError("cannot prune groups and in/out at the same time")

    if not hasattr(conv, "out_pruning_indices"):
        conv.out_pruning_indices = []
    conv.out_pruning_indices.append(idx)


def prune_group_conv(conv, idx):
    if hasattr(conv, "in_pruning_indices") or hasattr(conv, "out_pruning_indices"):
        raise RuntimeError("cannot prune groups and in/out at the same time")

    if not hasattr(conv, "group_pruning_indices"):
        conv.group_pruning_indices = []
    conv.group_pruning_indices.append(idx)


def prune_conv_in(conv, idx):
    if hasattr(conv, "group_pruning_indices"):
        raise RuntimeError("cannot prune groups and in/out at the same time")

    if not hasattr(conv, "in_pruning_indices"):
        conv.in_pruning_indices = []
    conv.in_pruning_indices.append(idx)


def prune_bn_multiple(bn, filters):
    keep_idxs = [idx for idx in range(bn.num_features) if idx not in filters]
    bn.num_features -= len(filters)

    bn.running_mean = select_at_dim(bn.running_mean, keep_idxs, 0)
    bn.running_var = select_at_dim(bn.running_var, keep_idxs, 0)
    bn.weight = nn.Parameter(select_at_dim(bn.weight.data.clone(), keep_idxs, 0))
    bn.bias = nn.Parameter(select_at_dim(bn.bias.data.clone(), keep_idxs, 0))


def prune_conv_out_multiple(conv, filters):
    keep_idxs = [idx for idx in range(conv.out_channels) if idx not in filters]
    conv.out_channels -= len(filters)

    conv.weight = nn.Parameter(select_at_dim(conv.weight.data.clone(), keep_idxs, 0))

    if conv.bias is not None:
        conv.bias = nn.Parameter(select_at_dim(conv.bias.data.clone(), keep_idxs, 0))


def prune_group_conv_multiple(conv, filters):
    keep_idxs = [idx for idx in range(conv.groups) if idx not in filters]
    conv.in_channels -= len(filters)
    conv.out_channels -= len(filters)
    conv.groups -= len(filters)

    conv.weight = nn.Parameter(select_at_dim(conv.weight.data.clone(), keep_idxs, 0))

    if conv.bias is not None:
        conv.bias = nn.Parameter(select_at_dim(conv.bias.data.clone(), keep_idxs, 0))


def prune_conv_in_multiple(conv, filters):
    keep_idxs = [idx for idx in range(conv.in_channels) if idx not in filters]
    conv.in_channels -= len(filters)
    conv.weight = nn.Parameter(select_at_dim(conv.weight.data.clone(), keep_idxs, 1))


def execute_pruning_steps(model):
    def _execute_conv2d(conv):
        if hasattr(conv, "in_pruning_indices"):
            prune_conv_in_multiple(conv, conv.in_pruning_indices)
            delattr(conv, "in_pruning_indices")
        if hasattr(conv, "out_pruning_indices"):
            prune_conv_out_multiple(conv, conv.out_pruning_indices)
            delattr(conv, "out_pruning_indices")
        if hasattr(conv, "group_pruning_indices"):
            prune_group_conv_multiple(conv, conv.group_pruning_indices)
            delattr(conv, "group_pruning_indices")

    def _execute_bn(bn):
        if hasattr(bn, "pruning_indices"):
            prune_bn_multiple(bn, bn.pruning_indices)
            delattr(bn, "pruning_indices")

    apply_to_type(model, nn.Conv2d, _execute_conv2d)
    apply_to_type(model, nn.BatchNorm2d, _execute_bn)


class PruningConf:
    def __init__(self, f=None, n_filters=-1, conv_statistics=None, readable_string=None):
        self.pruning_func = f
        self.n_filters = n_filters
        self.conv_statistics = conv_statistics if conv_statistics is not None else []
        self.readable_string = readable_string

    def reserve_execution(self, index):
        if not self.initialized():
            raise RuntimeError("not initialized")

        self.pruning_func(index)

    def get_stats(self):
        stats = np.zeros(self.n_filters)
        for s in self.conv_statistics:
            stats += s.filter_ranks

        # normalize for each block
        stats = np.abs(stats)
        stats = stats / np.sqrt(np.sum(stats * stats))
        return stats

    def initialized(self):
        return self.n_filters > -1 and self.pruning_func is not None
