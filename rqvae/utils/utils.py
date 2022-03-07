import random
import pickle

import numpy as np
import torch

from tqdm import tqdm
from torch.nn import functional as F


def save_pickle(fname, data):
    with open(fname, 'wb') as fp:
        pickle.dump(data, fp, pickle.HIGHEST_PROTOCOL)


def compute_p_norm(model):
    norm = 0
    for k, v in model.state_dict().items():
        v = v.detach().clone()
        norm += torch.sum(v.view(-1).pow_(2))
    return norm


def get_num_conv_linear_layers(model):
    cnt = 0
    weight_modules = (torch.nn.Linear, torch.nn.Conv2d, torch.nn.ConvTranspose2d)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            if pn.endswith('weight') and isinstance(m, weight_modules):
                cnt += 1
    return cnt


def compute_model_size(model, logger):
    if logger is not None:
        logger.info(
            "#parameters: %.4fM", sum(p.numel() for p in model.parameters()) / 1000 / 1000
        )


def set_seed(seed=None):
    if seed is None:
        seed = random.getrandbits(32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def np2tn(array):
    if len(array.shape) == 4:
        return torch.from_numpy(np.transpose(array, (3, 2, 0, 1)))
    elif len(array.shape) == 2:
        return torch.from_numpy(array.T)
    else:
        raise ValueError('invalid shape')


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out


def top_p_probs(probs, p):    
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)
    
    sorted_idx_remove_cond = cum_probs >= p
    
    sorted_idx_remove_cond[..., 1:] = sorted_idx_remove_cond[..., :-1].clone()
    sorted_idx_remove_cond[..., 0] = 0
    
    indices_to_remove = sorted_idx_remove_cond.scatter(-1, sorted_indices, sorted_idx_remove_cond)
    probs = probs.masked_fill(indices_to_remove, 0.0)
    norm_probs = probs / torch.sum(probs, dim=-1, keepdim=True)
    return norm_probs


def sample_from_logits(logits, temperature=1.0, top_k=None, top_p=None):
    """Take a 2-dim tensor, apply softmax along each row, and sample from
    each multinomial distribution defined by the rows.

    Args:
        logits: 2-dim tensor of shape (n_samples, logit_dim)
        temperature (float): softmax temperature
        top_k (Optional[int]): if given, sample only using `top_k` logits
        top_p (Optional[float]): if given, sample only using `top_p` logits

    Returns:
        samples: 1-dim integer tensor of shape (n_samples,)
    """

    logits = logits.to(dtype=torch.float32)
    logits = logits / temperature

    # optionally crop probabilities to only the top k options
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    
    if torch.sum(torch.isnan(logits)):
        print('WARNING... NaN observed')
        logits[torch.isnan(logits)] = -float('Inf')

    # apply softmax to convert to probabilities
    probs = F.softmax(logits, dim=-1)
    
    if top_p is not None:
        probs = top_p_probs(probs, top_p)

    try:
        samples = torch.multinomial(probs, num_samples=1)
    except RuntimeError:
        print(probs)
        print(logits)
        print('isinf, ', torch.sum(torch.isinf(probs)))
        print('isnan, ', torch.sum(torch.isnan(probs)))
        print('is negative', torch.sum(probs < 0))
        raise

    return samples.view(-1)
