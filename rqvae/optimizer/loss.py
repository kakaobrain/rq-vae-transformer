# Copyright (c) 2022-present, Kakao Brain Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import torch

import numpy as np
from torch.nn import functional as F

LOG_SCALE_MIN = -7


def compute_entropy(x, normalized=False):
    if not normalized:
        x /= np.sum(x)
    h = -np.sum(x * np.log(x + 1e-10))
    return h

def update_codebook_with_entropy(codebook, code):
    code_h, code_w = code.shape[1:]
    try:
        code = code.view(-1).cpu().numpy()
    except:
        code = code.view(-1).numpy()
    code, code_cnt = np.unique(code, return_counts=True)
    code_cnt = code_cnt.astype(np.float32) / (code_h*code_w)
    codebook[code] += code_cnt
    code_ent_ = compute_entropy(codebook)
    return codebook, code_ent_
    


def torch_compute_entropy(x, normalized=False):
    if not normalized:
        x = x / torch.sum(x, dim=-1, keepdim=True)
    h = -torch.sum(x * torch.log(x + 1e-10), dim=-1)
    return h


def to_one_hot(tensor, n, fill_with=1.):
    # we perform one hot encore with respect to the last axis
    one_hot = torch.FloatTensor(tensor.size() + (n,)).zero_()
    if tensor.is_cuda:
        one_hot = one_hot.cuda()
    one_hot.scatter_(len(tensor.size()), tensor.unsqueeze(-1), fill_with)
    return one_hot


def log_sum_exp(x, axis=1):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    # TF ordering -> NCHW format
    m, _ = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis) + 1e-7)


def log_prob_from_logits(x, axis=1):
    """ numerically stable log_softmax implementation that prevents overflow """
    # TF ordering -> NCHW format
    m, _ = torch.max(x, dim=axis, keepdim=True)
    return x - m - torch.log(torch.sum(torch.exp(x - m), dim=axis, keepdim=True) + 1e-7)


def soft_target_cross_entropy(input, target, reduction='mean'):
    loss = torch.sum(-target * log_prob_from_logits(input, axis=-1), dim=-1)
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError()
