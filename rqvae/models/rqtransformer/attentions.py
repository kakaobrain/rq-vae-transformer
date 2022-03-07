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
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F

from .configs import AttentionBlockConfig, AttentionStackConfig


class GELU(nn.Module):
    def __init__(self, version='v1'):
        super().__init__()
        assert version == 'v1' or version == 'v2'

        self.version = version

    def forward(self, x):
        if self.version == 'v1':
            return F.gelu(x)
        else:
            return x * torch.sigmoid(1.702 * x)


class MultiSelfAttention(nn.Module):
    """
    Optimized by batched matmul operations
    """

    def __init__(self, config: AttentionBlockConfig, mask=True):
        super().__init__()
        assert config.embed_dim % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.embed_dim, config.embed_dim, bias=config.attn_bias)
        self.query = nn.Linear(config.embed_dim, config.embed_dim, bias=config.attn_bias)
        self.value = nn.Linear(config.embed_dim, config.embed_dim, bias=config.attn_bias)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop, inplace=False)
        self.resid_drop = nn.Dropout(config.resid_pdrop, inplace=True)
        # output projection
        self.proj = nn.Linear(config.embed_dim, config.embed_dim, config.attn_bias)

        self.n_head = config.n_head
        self.mask = mask

    def forward(self, x, caching=False, past_kv=None):
        (B, T, C) = x.shape

        if not caching:
            assert past_kv is None

        x = x.transpose(0, 1).contiguous()  # (B, T, C) -> (T, B, C)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(T, B*self.n_head, C//self.n_head).transpose(0, 1)  # (B*nh, T, hs)
        q = self.query(x).view(T, B*self.n_head, C//self.n_head).transpose(0, 1)  # (B*nh, T, hs)
        v = self.value(x).view(T, B*self.n_head, C//self.n_head).transpose(0, 1)  # (B*nh, T, hs)

        if past_kv is not None:
            past_key, past_value = past_kv
            k = torch.cat([past_key, k], dim=-2)
            v = torch.cat([past_value, v], dim=-2)
            T_past = past_key.shape[1]
        else:
            T_past = 0

        if caching:
            present = torch.stack([k, v])
        else:
            present = None

        # Tensor shape below: (B * nh, T, hs) X (B * nh, hs, T_past+T) -> (B * nh, T, T_past+T)
        att = torch.bmm(q, (k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))))
        if self.mask:
            mask = torch.tril(torch.ones(T_past+T, T_past+T, device=x.device, dtype=torch.bool))
            mask = mask.view(1, T_past+T, T_past+T)
            att = att.masked_fill(~mask[:, T_past:T_past+T, :T_past+T], float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = torch.bmm(att, v)  # (B*nh, T, T_past+T) X (B*nh, T_past+T, hs) -> (B*nh, T, hs)
        y = y.transpose(0, 1).contiguous().view(T, B, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))

        if caching:
            return y.transpose(0, 1).contiguous(), present  # (T, B, C) -> (B, T, C)
        else:
            return y.transpose(0, 1).contiguous()  # (T, B, C) -> (B, T, C)


class AttentionBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config: AttentionBlockConfig):
        super().__init__()

        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.ln2 = nn.LayerNorm(config.embed_dim)

        self.attn = MultiSelfAttention(config, mask=True)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, 4 * config.embed_dim, bias=config.mlp_bias),
            GELU(config.gelu),
            nn.Linear(4 * config.embed_dim, config.embed_dim, bias=config.mlp_bias),
            nn.Dropout(config.resid_pdrop, inplace=True),
        )
        self._cache = None

    def forward(self, x):

        attn = self.attn(self.ln1(x))

        x = x + attn
        x = x + self.mlp(self.ln2(x))

        return x

    def cached_forward(self, x_present):

        attn, present = self.attn(self.ln1(x_present), caching=True, past_kv=self._cache['past_kv'])
        self._cache['past_kv'] = present

        x_present = x_present + attn
        x_present = x_present + self.mlp(self.ln2(x_present))

        return x_present

    def init_cache(self):
        self._cache = {'past_kv': None}


class AttentionStack(nn.Module):

    blocks: Iterable[AttentionBlock]

    def __init__(self, config: AttentionStackConfig):
        super().__init__()

        self.blocks = nn.ModuleList([AttentionBlock(config.block) for _ in range(config.n_layer)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

    def cached_forward(self, x_present):
        for block in self.blocks:
            x_present = block.cached_forward(x_present)
        return x_present

    def init_cache(self):
        for block in self.blocks:
            block.init_cache()
