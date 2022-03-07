"""Adapted and modified from https://github.com/CompVis/taming-transformers"""

import torch
import torch.nn.functional as F


def hinge_d_loss(logits_real, logits_fake, reduction='mean'):
    reduce_op = torch.mean if reduction == 'mean' else torch.sum
    loss_real = reduce_op(F.relu(1. - logits_real))
    loss_fake = reduce_op(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_g_loss(logits_fake, reduction='mean'):
    if reduction == 'mean':
        return -torch.mean(logits_fake)
    elif reduction == 'sum':
        return -torch.sum(logits_fake)


def vanilla_d_loss(logits_real, logits_fake, reduction='mean'):
    reduce_op = torch.mean if reduction == 'mean' else torch.sum
    d_loss = 0.5 * (
        reduce_op(torch.nn.functional.softplus(-logits_real)) +
        reduce_op(torch.nn.functional.softplus(logits_fake)))
    return d_loss
