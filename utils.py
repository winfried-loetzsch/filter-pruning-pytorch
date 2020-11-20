import torch
from torch.nn.functional import one_hot

smooth_const = 1e-6


def move_dim(t1, source, target):
    assert source != target
    offset = 0

    if 0 < source < target:
        offset = 1

    return t1.unsqueeze(target).transpose(target, offset + source).squeeze(offset + source)


def multiclass_iou_score(outputs, labels, n_classes=2, n_dims=2, smooth=smooth_const):
    axis = tuple(range(2, 2 + n_dims))  # spatial dimensions
    labels = move_dim(one_hot(labels, num_classes=n_classes), -1, 1)
    intersection = (outputs * labels).sum(axis)
    union = ((outputs + labels) - (outputs * labels)).sum(axis)
    iou = (intersection + smooth) / (union + smooth)

    # select the classes which are present in the data only
    # afterward calculate the mean over classes
    # otherwise results will be skewed
    select_m = torch.nn.functional.normalize(torch.clamp(labels.sum(axis), 0, 1.0).float(), p=1, dim=1)
    return torch.diagonal(select_m @ torch.transpose(iou, 0, 1)).mean()


def multiclass_dice_loss(outputs, labels, n_classes=2, n_dims=2, smooth=smooth_const):
    axis = tuple(range(2, 2 + n_dims))  # spatial dimensions
    labels_oh = move_dim(one_hot(labels, num_classes=n_classes), -1, 1).float()
    numerator = 2 * (outputs * labels_oh).sum(axis)
    denominator = (torch.square(outputs) + torch.square(labels_oh)).sum(axis)  # no square?

    dice = (numerator + smooth) / (denominator + smooth)
    return 1 - dice.mean()


class IOUMultiClassMetric:
    def __init__(self, nclasses):
        self.nclasses = nclasses

    def __call__(self, ypred, yhat, *args, **kwargs):
        ypred = ypred.argmax(1)
        ypred = move_dim(one_hot(ypred, num_classes=self.nclasses), -1, 1)

        return multiclass_iou_score(ypred, yhat, self.nclasses).item()
