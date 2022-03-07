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

import abc

from torch import nn


class Stage1Model(nn.Module, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get_codes(self, *args, **kwargs):
        """Generate the code from the input."""
        pass

    @abc.abstractmethod
    def decode_code(self, *args, **kwargs):
        """Generate the decoded image from the given code."""
        pass

    @abc.abstractmethod
    def get_recon_imgs(self, *args, **kwargs):
        """Scales the real and recon images properly.
        """
        pass

    @abc.abstractmethod
    def compute_loss(self, *args, **kwargs):
        """Compute the losses necessary for training.

        return {
            'loss_total': ...,
            'loss_recon': ...,
            'loss_latent': ...,
            'codes': ...,
            ...
        }
        """
        pass


class Stage2Model(nn.Module, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def compute_loss(self, *args, **kwargs):
        """Compute the losses necessary for training.
        Typically, it would be the cross-entropy of the AR prediction w.r.t. the ground truth.
        """
        pass

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_block_size(self):
        return self.block_size
