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

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..interfaces import Stage1Model
from .quantizations import RQBottleneck
from .modules import Encoder, Decoder
from .layers import ResnetBlock


class RQVAE(Stage1Model):
    def __init__(self,
                 *,
                 embed_dim=64,
                 n_embed=512,
                 decay=0.99,
                 loss_type='mse',
                 latent_loss_weight=0.25,
                 bottleneck_type='rq',
                 ddconfig=None,
                 checkpointing=False,
                 **kwargs):
        super().__init__()

        assert loss_type in ['mse', 'l1']

        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)

        def set_checkpointing(m):
            if isinstance(m, ResnetBlock):
                m.checkpointing = checkpointing

        self.encoder.apply(set_checkpointing)
        self.decoder.apply(set_checkpointing)

        if bottleneck_type == 'rq':
            latent_shape = kwargs['latent_shape']
            code_shape = kwargs['code_shape']
            shared_codebook = kwargs['shared_codebook']
            restart_unused_codes = kwargs['restart_unused_codes']
            self.quantizer = RQBottleneck(latent_shape=latent_shape,
                                          code_shape=code_shape,
                                          n_embed=n_embed,
                                          decay=decay,
                                          shared_codebook=shared_codebook,
                                          restart_unused_codes=restart_unused_codes,
                                          )
            self.code_shape = code_shape
        else:
            raise ValueError("invalid 'bottleneck_type' (must be 'rq')")

        self.quant_conv = nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

        self.loss_type = loss_type
        self.latent_loss_weight = latent_loss_weight

    def forward(self, xs):
        z_e = self.encode(xs)
        z_q, quant_loss, code = self.quantizer(z_e)
        out = self.decode(z_q)
        return out, quant_loss, code

    def encode(self, x):
        z_e = self.encoder(x)
        z_e = self.quant_conv(z_e).permute(0, 2, 3, 1).contiguous()
        return z_e

    def decode(self, z_q):
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        z_q = self.post_quant_conv(z_q)
        out = self.decoder(z_q)
        return out

    @torch.no_grad()
    def get_codes(self, xs):
        z_e = self.encode(xs)
        _, _, code = self.quantizer(z_e)
        return code

    @torch.no_grad()
    def get_soft_codes(self, xs, temp=1.0, stochastic=False):
        assert hasattr(self.quantizer, 'get_soft_codes')

        z_e = self.encode(xs)
        soft_code, code = self.quantizer.get_soft_codes(z_e, temp=temp, stochastic=stochastic)
        return soft_code, code

    @torch.no_grad()
    def decode_code(self, code):
        z_q = self.quantizer.embed_code(code)
        decoded = self.decode(z_q)
        return decoded

    def get_recon_imgs(self, xs_real, xs_recon):

        xs_real = xs_real * 0.5 + 0.5
        xs_recon = xs_recon * 0.5 + 0.5
        xs_recon = torch.clamp(xs_recon, 0, 1)

        return xs_real, xs_recon

    def compute_loss(self, out, quant_loss, code, xs=None, valid=False):

        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(out, xs, reduction='mean')
        elif self.loss_type == 'l1':
            loss_recon = F.l1_loss(out, xs, reduction='mean')
        else:
            raise ValueError('incompatible loss type')

        loss_latent = quant_loss

        if valid:
            loss_recon = loss_recon * xs.shape[0] * xs.shape[1]
            loss_latent = loss_latent * xs.shape[0]

        loss_total = loss_recon + self.latent_loss_weight * loss_latent

        return {
            'loss_total': loss_total,
            'loss_recon': loss_recon,
            'loss_latent': loss_latent,
            'codes': [code]
        }

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def get_code_emb_with_depth(self, code):
        return self.quantizer.embed_code_with_depth(code)

    @torch.no_grad()
    def decode_partial_code(self, code, code_idx, decode_type='select'):
        r"""
        Use partial codebooks and decode the codebook features.
        If decode_type == 'select', the (code_idx)-th codebook features are decoded.
        If decode_type == 'add', the [0,1,...,code_idx]-th codebook features are added and decoded.
        """
        z_q = self.quantizer.embed_partial_code(code, code_idx, decode_type)
        decoded = self.decode(z_q)
        return decoded

    @torch.no_grad()
    def forward_partial_code(self, xs, code_idx, decode_type='select'):
        r"""
        Reconstuct an input using partial codebooks.
        """
        code = self.get_codes(xs)
        out = self.decode_partial_code(code, code_idx, decode_type)
        return out
