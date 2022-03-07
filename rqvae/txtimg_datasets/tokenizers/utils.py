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

import os
from functools import lru_cache


@lru_cache()
def default_bpe():
    # used in the original CLIP implementation
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "pretrained",
                        "bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bert_vocab():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "pretrained",
                        "bert-base-uncased-vocab.txt")


@lru_cache()
def gpt2_vocab():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "pretrained",
                        "vocab.json")


@lru_cache()
def gpt2_merges():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "pretrained",
                        "merges.txt")


@lru_cache()
def huggingface_bpe_16k_vocab():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "pretrained",
                        "bpe-16k-vocab.json")


@lru_cache()
def huggingface_bpe_16k_merges():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "pretrained",
                        "bpe-16k-merges.txt")


@lru_cache()
def huggingface_bpe_30k_vocab():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "pretrained",
                        "bpe-30k-vocab.json")


@lru_cache()
def huggingface_bpe_30k_merges():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "pretrained",
                        "bpe-30k-merges.txt")
