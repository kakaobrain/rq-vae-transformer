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

from typing import List, Optional, Any
from dataclasses import dataclass

from omegaconf import OmegaConf, MISSING


@dataclass
class AttentionBlockConfig:
    embed_dim: int = MISSING
    n_head: int = MISSING
    mlp_bias: bool = True
    attn_bias: bool = True
    attn_pdrop: float = 0.0
    resid_pdrop: float = 0.1
    gelu: str = 'v1'


@dataclass
class AttentionStackConfig:
    n_layer: int = MISSING
    block: AttentionBlockConfig = AttentionBlockConfig()


@dataclass
class RQTransformerConfig:

    type: str = 'rq-transformer'
    ema: Optional[bool] = None
    ar_hierarchy: Optional[bool] = None

    vocab_size: Any = MISSING
    block_size: List[int] = MISSING

    vocab_size_cond: int = 0
    block_size_cond: int = 0

    embed_dim: int = MISSING
    input_embed_dim: Optional[int] = None
    use_padding_emb: bool = False

    input_emb_vqvae: bool = False
    head_emb_vqvae: bool = False
    scaled_head_emb_vqvae: bool = False
    cumsum_depth_ctx: bool = False
    shared_tok_emb: bool = False

    embd_pdrop: float = 0.0

    body: AttentionStackConfig = AttentionStackConfig()
    head: AttentionStackConfig = AttentionStackConfig()

    shared_cls_emb: bool = False

    @classmethod
    def create(cls, config):
        defaults = OmegaConf.structured(cls(embed_dim=config.embed_dim))
        defaults.body.block.embed_dim = defaults.embed_dim
        defaults.head.block.embed_dim = defaults.embed_dim
        return OmegaConf.merge(defaults, config)
