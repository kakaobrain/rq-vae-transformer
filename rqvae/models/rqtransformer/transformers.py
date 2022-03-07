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

from collections import OrderedDict
from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from tqdm import tqdm

from rqvae.utils.utils import sample_from_logits
from rqvae.optimizer.loss import soft_target_cross_entropy

from ..interfaces import Stage2Model
from .primitives import BatchLinear, TupleEmbedding, LogitMask
from .attentions import AttentionStack
from .configs import RQTransformerConfig


class RQTransformer(Stage2Model):

    def __init__(self, config: RQTransformerConfig):
        super().__init__()

        self.config = config = config.copy()

        if len(config.block_size) != 3:
            raise ValueError("incompatible block size")
        self.block_size = torch.Size(config.block_size)

        if isinstance(config.vocab_size, int):
            config.vocab_size = [config.vocab_size] * config.block_size[2]

        if config.shared_tok_emb or config.shared_cls_emb:
            # various codebooks sizes are not supported for shared tok or cls embedding
            assert [config.vocab_size[0]] * len(config.vocab_size) == config.vocab_size

        self.vocab_size = config.vocab_size

        # ==== embedding layer definitions ====

        # vocab_size_cond == 1 => cond_emb works as a SOS token provider
        self.vocab_size_cond = max(config.vocab_size_cond, 1)
        self.block_size_cond = max(config.block_size_cond, 1)
        assert not (self.block_size_cond > 1 and self.vocab_size_cond == 1)
        self.cond_emb = nn.Embedding(self.vocab_size_cond, config.embed_dim)

        self.tok_emb, self.input_mlp, self.head_mlp = None, None, None
        if config.input_emb_vqvae:
            self.input_mlp = nn.Linear(config.input_embed_dim, config.embed_dim)

        if config.head_emb_vqvae:
            self.head_mlp = nn.Linear(config.input_embed_dim, config.embed_dim)

        if not (config.input_emb_vqvae and config.head_emb_vqvae):
            if config.shared_tok_emb:
                self.tok_emb = nn.Embedding(config.vocab_size[0], config.embed_dim)
            else:
                self.tok_emb = TupleEmbedding(config.vocab_size, config.embed_dim)

        self.pos_emb_cond = nn.Parameter(torch.zeros(1, self.block_size_cond, config.embed_dim))
        self.pos_emb_hw = nn.Parameter(torch.zeros(1, self.block_size[0] * self.block_size[1], config.embed_dim))
        self.pos_emb_d = nn.Parameter(torch.zeros(1, self.block_size[2], config.embed_dim))

        self.pos_emb_cond.data.normal_(mean=0.0, std=0.02)
        self.pos_emb_hw.data.normal_(mean=0.0, std=0.02)
        self.pos_emb_d.data.normal_(mean=0.0, std=0.02)

        self.embed_drop = nn.Dropout(config.embd_pdrop, inplace=True)

        # ==== AR modeling layer definitions ====
        self.body_transformer = AttentionStack(config.body)
        self.head_transformer = AttentionStack(config.head)

        # ==== final fc layer definition ====
        self.classifier = nn.Sequential(OrderedDict([
            ('layer_norm', nn.LayerNorm(config.embed_dim)),
            (
                'linear',
                nn.Linear(config.embed_dim, config.vocab_size[0])
                if config.shared_cls_emb else
                BatchLinear(config.block_size[2], config.embed_dim, max(config.vocab_size))
            ),
            ('logit_mask', LogitMask(config.vocab_size, value=-1e6))
        ]))

        if config.block_size_cond > 1:
            self.cond_classifier = nn.Sequential(OrderedDict([
                ('layer_norm', nn.LayerNorm(config.embed_dim)),
                ('linear', nn.Linear(config.embed_dim, config.vocab_size_cond)),
            ]))

        self._cache = None

    def embed_with_model_aux(self, xs, model_aux):
        xs_emb, _ = model_aux.get_code_emb_with_depth(xs)
        return xs_emb

    def forward(self, xs, model_aux=None, cond=None, amp=False):
        with autocast(enabled=amp):

            (B, H, W, D) = xs.shape

            xs = xs.reshape(B, H*W, D)
            if cond is None:
                cond = torch.zeros(B, self.block_size_cond, device=xs.device, dtype=torch.long)
            else:
                cond = cond.reshape(B, self.block_size_cond)

            seq_len = xs.shape[1]
            cond_len = cond.shape[1]

            # compute the embeddings for body
            if self.config.input_emb_vqvae:
                xs_emb = self.embed_with_model_aux(xs, model_aux)
                xs_emb = self.input_mlp(xs_emb)
            else:
                xs_emb = self.tok_emb(xs)

            conds_emb = self.cond_emb(cond) + self.pos_emb_cond[:, :cond_len, :]
            xs_emb = xs_emb.sum(dim=-2) + self.pos_emb_hw[:, :seq_len, :]
            latents = torch.cat(
                [
                    conds_emb,
                    xs_emb[:, :-1, :]
                ],
                dim=1,
            )
            # NOTE: dropout applied after everything is combined, not as before
            latents = self.embed_drop(latents)

            # body transformer
            latents = self.body_transformer(latents)
            spatial_ctx = latents[:, cond_len-1:]

            # if cond_len > 1, compute the logits for conditioning sequence.
            if cond_len > 1:
                cond_ctx = latents[:, :cond_len-1]
                cond_logits = self.cond_classifier(cond_ctx)

            # compute the embeddings for head
            if self.config.head_emb_vqvae:
                depth_ctx = self.embed_with_model_aux(xs, model_aux)

                if self.config.cumsum_depth_ctx:
                    depth_ctx = torch.cumsum(depth_ctx, dim=-2)

                depth_ctx = self.head_mlp(depth_ctx)
            else:
                depth_ctx = self.tok_emb(xs)

            # NOTE: We are no longer applying spatial positional embedding to depth_ctx.
            # depth_ctx = depth_ctx + self.pos_emb_hw[:, :seq_len, :]

            depth_ctx_full = torch.cat(
                [
                    spatial_ctx.view(B, seq_len, 1, -1),
                    depth_ctx[:, :, :-1, :],
                ],
                dim=-2,
            )
            depth_ctx_full = depth_ctx_full.reshape(B * seq_len, D, -1)
            depth_ctx_full = depth_ctx_full + self.pos_emb_d[:, :D, :]

            # head transformer & final fc (classifier)
            head_outputs = self.head_transformer(depth_ctx_full)
            head_outputs = head_outputs.reshape(B, H, W, D, -1)

            seq_logits = self.classifier(head_outputs)

            if cond_len > 1:
                return seq_logits, cond_logits  # shape: (B, H, W, D, vocab_size), (B, cond_len-1, vocab_size_cond)
            else:
                return seq_logits

    @torch.no_grad()
    def cached_forward(self, xs, model_aux=None, cond=None, amp=False, sample_loc=(0, 0, 0)):
        """
        What should be the shape of xs?
            - just full tensor, we will slice with respect to the sample_loc

        What should be the shape of the output?
            - (B, vocab_size)
        """
        (h, w, d) = sample_loc
        (B, H, W, D) = xs.shape

        sampling_idx = h * W + w
        xs = xs.clone().reshape(B, -1, D)
        xs = xs[:, :sampling_idx+1]

        with autocast(enabled=amp):

            if cond is None:
                cond = torch.zeros(B, self.block_size_cond, device=xs.device, dtype=torch.long)
            else:
                cond = cond.reshape(B, self.block_size_cond)

            seq_len = xs.shape[1]
            cond_len = cond.shape[1]

            if d == 0:
                # Computing embedding for full input is wasteful, but code is simpler...
                if self.config.input_emb_vqvae:
                    xs_emb = self.embed_with_model_aux(xs, model_aux)
                    xs_emb = self.input_mlp(xs_emb)
                else:
                    xs_emb = self.tok_emb(xs)

                conds_emb = self.cond_emb(cond) + self.pos_emb_cond[:, :cond_len, :]
                xs_emb = xs_emb.sum(dim=-2) + self.pos_emb_hw[:, :seq_len, :]
                latents = torch.cat(
                    [
                        conds_emb,
                        xs_emb[:, :-1, :]
                    ],
                    dim=1,
                )
                latents = self.embed_drop(latents)

                latents_present = latents[:, :cond_len+sampling_idx, :]

                if self._cache['spatial_ctx_hw'] is None:
                    latents_present = self.body_transformer.cached_forward(latents_present)
                    spatial_ctx_hw = latents_present[:, -1, :].unsqueeze(1)
                else:
                    latents_hw = latents_present[:, -1, :].unsqueeze(1)
                    spatial_ctx_hw = self.body_transformer.cached_forward(latents_hw)

                self._cache['spatial_ctx_hw'] = spatial_ctx_hw

            spatial_ctx_hw = self._cache['spatial_ctx_hw']

            # compute the embeddings for head
            if self.config.head_emb_vqvae:
                depth_ctx = self.embed_with_model_aux(xs, model_aux)

                if self.config.cumsum_depth_ctx:
                    depth_ctx = torch.cumsum(depth_ctx, dim=-2)

                depth_ctx = self.head_mlp(depth_ctx)
            else:
                depth_ctx = self.tok_emb(xs)

            depth_ctx_hw = depth_ctx[:, sampling_idx, :]
            depth_ctx_full_hw = torch.cat(
                [
                    spatial_ctx_hw.view(B, 1, -1),
                    depth_ctx_hw[:, :-1, :],
                ],
                dim=-2,
            )
            depth_ctx_full_hw = depth_ctx_full_hw + self.pos_emb_d[:, :D, :]

            # head transformer & final fc (classifier)
            depth_ctx_full_hwd = depth_ctx_full_hw[:, d, :].unsqueeze(1)
            if d == 0:
                self.head_transformer.init_cache()
            head_outputs_hwd = self.head_transformer.cached_forward(depth_ctx_full_hwd)

            # head_outputs_hw = self.head_transformer(depth_ctx_full_hw)
            # head_outputs_hwd = head_outputs_hw[:, d, :].unsqueeze(1)

            if self.config.shared_cls_emb:
                logits_hwd = self.classifier(head_outputs_hwd)
            else:
                logits_hwd = self.classifier.layer_norm(head_outputs_hwd)
                logits_hwd = self.classifier.linear(logits_hwd, indices=[d])
                logits_hwd = self.classifier.logit_mask(logits_hwd)

            logits_hwd = logits_hwd.reshape(B, -1)

            return logits_hwd

    def init_cache(self):
        self._cache = {'spatial_ctx_hw': None}
        self.body_transformer.init_cache()
        self.head_transformer.init_cache()

    @torch.no_grad()
    def sample(self,
               partial_sample,
               model_aux=None,
               cond=None,
               start_loc=(0, 0),
               temperature=1.0,
               top_k=None,
               top_p=None,
               amp=False,
               cached=True,
               is_tqdm=False,
               desc="Sampling",
               fast=True,
               ):

        assert self.block_size == partial_sample.shape[1:]

        (H, W, D) = self.block_size

        if top_k is None:
            top_k_list = [self.vocab_size[i] for i in range(D)]
        elif isinstance(top_k, int):
            top_k_list = [min(top_k, self.vocab_size[i]) for i in range(D)]
        elif len(top_k) == 1:
            top_k_list = [min(top_k[0], self.vocab_size[i]) for i in range(D)]
        else:
            top_k_list = [min(top_k[i], self.vocab_size[i]) for i in range(D)]

        if top_p is None:
            top_p_list = [1.0 for _ in range(D)]
        elif isinstance(top_p, float):
            top_p_list = [min(top_p, 1.0) for _ in range(D)]
        elif len(top_p) == 1:
            top_p_list = [min(top_p[0], 1.0) for _ in range(D)]
        else:
            top_p_list = [min(top_p[i], 1.0) for i in range(D)]

        xs = partial_sample.clone()
        assert xs.shape[1:] == torch.Size([H, W, D])

        sample_locs = list(product(range(H), range(W), range(D)))

        if is_tqdm:
            pbar = tqdm(sample_locs, total=len(sample_locs))
            pbar.set_description(desc)
        else:
            pbar = sample_locs

        self.init_cache()

        for (h, w, d) in pbar:

            if (h, w) < (start_loc[0], start_loc[1]):
                continue

            xs_partial = xs[:, :h + 1]

            if cached:
                logits_hwd = self.cached_forward(xs_partial, model_aux, cond=cond, amp=amp, sample_loc=(h, w, d))
            else:
                logits = self(xs_partial, model_aux, cond=cond, amp=amp)
                logits_hwd = logits[:, h, w, d]
            
            _top_k = top_k_list[d] 
            _top_p = top_p_list[d]             

            samples_hwd = sample_from_logits(logits_hwd,
                                             temperature=temperature,
                                             top_k=_top_k,
                                             top_p=_top_p)
            xs[:, h, w, d] = samples_hwd

        self.init_cache()

        return xs

    def compute_loss(self, logits, targets, use_soft_target=False):

        logits = logits.reshape(-1, logits.shape[-1])
        if use_soft_target:
            targets = targets.reshape(-1, targets.shape[-1])
            loss = soft_target_cross_entropy(logits, targets)
        else:
            targets = targets.reshape(-1)
            loss = F.cross_entropy(logits, targets)

        return loss

    def compute_cond_loss(self, cond_logits, conds):
        assert cond_logits.shape[1] == (conds.shape[1] - 1)

        targets = conds[:, 1:].contiguous()
        cond_loss = F.cross_entropy(
            cond_logits.reshape(-1, cond_logits.shape[-1]),
            targets.reshape(-1)
        )
        return cond_loss

    @torch.no_grad()
    def compute_codebook_loss(self, logits, targets, use_soft_target=False):
        """Compute xent loss of each codebook for logging"""
        num_codebook = self.block_size[-1]
        (B, H, W, D, _) = logits.shape

        logits = logits.reshape(-1, logits.shape[-1])
        if use_soft_target:
            logits = logits.reshape(-1, logits.shape[-1])
            targets = targets.reshape(-1, targets.shape[-1])
            tokenwise_loss = soft_target_cross_entropy(logits, targets, reduction='none')
        else:
            targets = targets.reshape(-1)
            tokenwise_loss = F.cross_entropy(logits, targets, reduction='none')

        codebook_loss = tokenwise_loss.reshape(-1, D).mean(dim=0)

        return codebook_loss
