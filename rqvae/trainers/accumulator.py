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

from typing import Iterable

import torch

import rqvae.utils.dist as dist_utils
from rqvae.optimizer.loss import torch_compute_entropy


def assign_code(codebook, code):

    if len(code.shape) == 3:
        code = code.reshape(*code.shape, 1)

    n_codebooks = codebook.shape[0]
    code_h, code_w, code_d = code.shape[1:]
    chunks = torch.chunk(code, chunks=n_codebooks, dim=-1)
    for i, chunk in enumerate(chunks):
        uniques, counts = torch.unique(chunk.view(-1), return_counts=True)
        freqs = counts.to(dtype=torch.float) / (code_h * code_w)
        codebook[i][uniques] += freqs


class SummaryStage1:
    def __init__(self, loss_total, loss_recon, loss_latent, ent_codes_w_pad, ent_codes_wo_pad):
        self.loss_total = loss_total
        self.loss_recon = loss_recon
        self.loss_latent = loss_latent
        self.ent_codes_w_pad = ent_codes_w_pad
        self.ent_codes_wo_pad = ent_codes_wo_pad

    def print_line(self):
        loss_total = self.loss_total.item()
        loss_recon = self.loss_recon.item()
        loss_latent = self.loss_latent.item()

        line = f"loss_total: {loss_total:.4f}, loss_recon: {loss_recon:.4f}, loss_latent: {loss_latent:.4f}, "

        if self.ent_codes_w_pad is not None:
            for level, ent_code in enumerate(self.ent_codes_w_pad):
                ent_code_str = '[' + ', '.join([f'{ent:.4f}' for ent in ent_code]) + ']'
                line += f"""w/ pad entropy-level-{level}: {ent_code_str}, """

        for level, ent_code in enumerate(self.ent_codes_wo_pad):
            ent_code_str = '[' + ', '.join([f'{ent:.4f}' for ent in ent_code]) + ']'
            line += f"""w/o pad entropy-level-{level}: {ent_code_str}, """

        return line

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)


class AccmStage1:
    def __init__(self, n_codebook=1, codebook_size=512, code_hier=1, use_padding_idx=False, device='cpu'):
        self.n_codebook = n_codebook
        self.max_codebook_size = self.codebook_size = codebook_size
        self.use_padding_idx = use_padding_idx

        if isinstance(codebook_size, Iterable):
            self.max_codebook_size = max(codebook_size)

        if self.use_padding_idx:
            self.max_codebook_size += 1

        self.code_hier = code_hier
        self.device = device

        self.init()

    def init(self):
        self.loss_total = torch.zeros(1, device=self.device)
        self.loss_recon = torch.zeros(1, device=self.device)
        self.loss_latent = torch.zeros(1, device=self.device)

        self.codebooks = [torch.zeros(self.n_codebook, self.max_codebook_size, device=self.device)
                          for _ in range(self.code_hier)]
        self.counter = 0

    @torch.no_grad()
    def update(self,
               loss_total,
               loss_recon,
               loss_latent,
               codes,
               count=None,
               sync=False,
               distenv=None):

        if sync:
            loss_total = dist_utils.all_gather_cat(distenv, loss_total.unsqueeze(0)).sum()
            loss_recon = dist_utils.all_gather_cat(distenv, loss_recon.unsqueeze(0)).sum()
            loss_latent = dist_utils.all_gather_cat(distenv, loss_latent.unsqueeze(0)).sum()
            codes = [dist_utils.all_gather_cat(distenv, code) for code in codes]

        self.loss_total += loss_total.detach()
        self.loss_recon += loss_recon.detach()
        self.loss_latent += loss_latent.detach()

        for i in range(self.code_hier):
            assign_code(self.codebooks[i], codes[i].detach())

        self.counter += count if not sync else count * distenv.world_size

    @torch.no_grad()
    def get_summary(self, n_samples=None):
        n_samples = n_samples if n_samples else self.counter

        loss_total = self.loss_total / n_samples
        loss_recon = self.loss_recon / n_samples
        loss_latent = self.loss_latent / n_samples

        if self.use_padding_idx:
            ent_codes_w_pad = [torch_compute_entropy(self.codebooks[i]) for i in range(self.code_hier)]
            ent_codes_wo_pad = [torch_compute_entropy(self.codebooks[i][:, :-1]) for i in range(self.code_hier)]
        else:
            ent_codes_w_pad = None
            ent_codes_wo_pad = [torch_compute_entropy(self.codebooks[i][:, :-1]) for i in range(self.code_hier)]

        summary = SummaryStage1(loss_total=loss_total,
                                loss_recon=loss_recon,
                                loss_latent=loss_latent,
                                ent_codes_w_pad=ent_codes_w_pad,
                                ent_codes_wo_pad=ent_codes_wo_pad)

        return summary


class SummaryStage1WithGAN:
    def __init__(self, ent_codes_w_pad, ent_codes_wo_pad, **kwargs):
        for k, v in kwargs.items():
            self[k] = v
        self.ent_codes_w_pad = ent_codes_w_pad
        self.ent_codes_wo_pad = ent_codes_wo_pad

    def print_line(self):
        line = ""
        for name, value in self.metrics.items():
            line += f"{name}: {value.item():.4f}, "

        if self.ent_codes_w_pad is not None:
            for level, ent_code in enumerate(self.ent_codes_w_pad):
                ent_code_str = '[' + ', '.join([f'{ent:.4f}' for ent in ent_code]) + ']'
                line += f"""w/ pad entropy-level-{level}: {ent_code_str}, """

        for level, ent_code in enumerate(self.ent_codes_wo_pad):
            ent_code_str = '[' + ', '.join([f'{ent:.4f}' for ent in ent_code]) + ']'
            line += f"""w/o pad entropy-level-{level}: {ent_code_str}"""

        return line

    @property
    def metrics(self):
        def is_scalar(value):
            return (isinstance(value, torch.Tensor) and value.numel() == 1) or isinstance(value, float)

        return {key: value for key, value in self.__dict__.items() if is_scalar(value)}

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)


class AccmStage1WithGAN:
    def __init__(self, metric_names, n_codebook=1, codebook_size=512, code_hier=1, use_padding_idx=False, device='cpu'):
        self.n_codebook = n_codebook
        self.max_codebook_size = self.codebook_size = codebook_size
        self.use_padding_idx = use_padding_idx

        if isinstance(codebook_size, list):
            self.max_codebook_size = max(codebook_size)

        if self.use_padding_idx:
            self.max_codebook_size += 1

        self.code_hier = code_hier
        self.device = device

        self.metrics_sum = {n: torch.zeros(1, device=self.device) for n in metric_names}

        self.codebooks = [torch.zeros(self.n_codebook, self.max_codebook_size, device=self.device)
                          for _ in range(self.code_hier)]
        self.counter = 0

    @torch.no_grad()
    def update(self,
               codes,
               metrics_to_add,
               count=None,
               sync=False,
               distenv=None):

        if sync:
            codes = [dist_utils.all_gather_cat(distenv, code) for code in codes]
            for name, value in metrics_to_add.items():
                gathered_value = dist_utils.all_gather_cat(distenv, value.unsqueeze(0))
                gathered_value = gathered_value.sum().detach()
                metrics_to_add[name] = gathered_value

        for name, value in metrics_to_add.items():
            if name not in self.metrics_sum:
                raise KeyError(f'unexpected metric name: {name}')
            self.metrics_sum[name] += value

        for i in range(self.code_hier):
            assign_code(self.codebooks[i], codes[i].detach())

        self.counter += count if not sync else count * distenv.world_size

    @torch.no_grad()
    def get_summary(self, n_samples=None):
        n_samples = n_samples if n_samples else self.counter

        metrics_avg = {k: v / n_samples for k, v in self.metrics_sum.items()}

        if self.use_padding_idx:
            ent_codes_w_pad = [torch_compute_entropy(self.codebooks[i]) for i in range(self.code_hier)]
            ent_codes_wo_pad = [torch_compute_entropy(self.codebooks[i][:, :-1]) for i in range(self.code_hier)]
        else:
            ent_codes_w_pad = None
            ent_codes_wo_pad = [torch_compute_entropy(self.codebooks[i]) for i in range(self.code_hier)]

        summary = SummaryStage1WithGAN(ent_codes_w_pad=ent_codes_w_pad,
                                       ent_codes_wo_pad=ent_codes_wo_pad,
                                       **metrics_avg)

        return summary
