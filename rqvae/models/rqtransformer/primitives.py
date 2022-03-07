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

from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

Size = Union[Tuple[int, ...], List[int], torch.Size]


class TupleEmbedding(nn.Embedding):
    r"""A simple lookup table that stores embeddings of multiple dictionaries and fixed size.

    This module intends to represent a tuple (x_1, ..., x_k) from the product of k (possibly differently-sized)
    dictionaries by the tuple of the embeddings of individual entries.
    The input to the module is a list of tuples of indices, and the output is the corresponding
    tuple embeddings.

    Args:
        num_embeddings (int or tuple): list of the sizes of each dictionary of embeddings
        embedding_dim (int): the size of each embedding vector

    Shape:
        - Input: :math:`(*, D)`, IntTensor or LongTensor of arbitrary shape containing the indices to extract
        - Output: :math:`(*, D, H)`, where `*` is the input shape and :math:`H=\text{embedding\_dim}`
    """

    def __init__(self, num_embeddings: Union[int, Iterable[int]], embedding_dim, **kwargs) -> None:

        if 'padding_idx' in kwargs:
            raise ValueError('padding_idx argument not supported')

        if isinstance(num_embeddings, int):
            num_embeddings = (num_embeddings,)

        self.num_embeddings_per_dict = num_embeddings
        self.embedding_dim = embedding_dim

        super(TupleEmbedding, self).__init__(num_embeddings=sum(self.num_embeddings_per_dict),
                                             embedding_dim=embedding_dim,
                                             **kwargs)

        self.register_buffer('offsets', None)
        self.offsets = torch.tensor(np.cumsum([0] + self.num_embeddings_per_dict[:-1]), dtype=torch.long)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        (*rem, D) = x.shape
        assert D == len(self.num_embeddings_per_dict)

        offsets = self.offsets.view(*[1 for _ in rem], D)
        x_emb = super(TupleEmbedding, self).forward(x + offsets)

        return x_emb


class LogitMask(nn.Module):
    def __init__(self, vocab_size: Iterable[int], value=-1e6):
        super().__init__()

        self.vocab_size = vocab_size
        self.mask_cond = [vocab_size[0]]*len(vocab_size) != vocab_size
        self.value = value

    def forward(self, logits: Tensor) -> Tensor:
        if not self.mask_cond:
            return logits
        else:
            for idx, vocab_size in enumerate(self.vocab_size):
                logits[:, idx, vocab_size:].fill_(-float('Inf'))
            return logits


class BatchLinear(nn.Module):
    r"""Applies multiple linear transformations to multiple vectors in a batched way:

    .. math::
        y_i = x_i A_i^T + b_i \text{for } i=1, \cdots, n_{vectors}

    Args:
        n_vectors (int): number of linear transformations (=number of vectors in input)
        in_features (int): size of each input sample
        out_features (int): size of each output sample
        bias (bool): If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
    """
    bias: Optional[Tensor]

    def __init__(self, n_vectors: int, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.n_vectors = n_vectors
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.Tensor(n_vectors, in_features, out_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_vectors, out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.weight.data.normal_(mean=0.0, std=0.02)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input: Tensor, indices=None) -> Tensor:
        """
        Inputs:
            input (Tensor): A tensor to which linear transfs. are applied
            indices (optional, List[int]): List of indices of linear transf. to be applied in a batched manner.
                If 'None', all linear transforms are applied.
        Shapes:
            - input: (*, n_vectors, in_channel)
            - output: (*, n_vectors, out_channel)
        Output:
            Tensor(shape=[..., n_vectors, out_channel])
        """
        (*rem, n_vectors, in_ch) = input.shape

        if indices:
            assert n_vectors == len(indices)
            weight = self.weight[indices]
            if self.bias is not None:
                bias = self.bias[indices]
            else:
                bias = None
        else:
            weight = self.weight
            bias = self.bias

        output = torch.einsum('bij,ijk->bik',
                              input.view(-1, n_vectors, in_ch),
                              weight,
                              )

        if bias is not None:
            output = output + bias.unsqueeze(0)

        return output.reshape(*rem, n_vectors, -1)

    def extra_repr(self) -> str:
        return 'n_vectors={}, in_features={}, out_features={}, bias={}'.format(
            self.n_vectors, self.in_features, self.out_features, self.bias is not None
        )
