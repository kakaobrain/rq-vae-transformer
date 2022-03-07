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

import torch
import torch.utils.data
from torchvision.datasets import VisionDataset
from PIL import Image
from tqdm import tqdm

from .tokenizers import create_tokenizer


class Cc3m(VisionDataset):
    splits = {'train', 'val'}

    def __init__(self, root, split, tok_name, transform=None, context_length=77, dropout=None):
        assert split in self.splits, f'{split} is not in {self.splits}'
        super().__init__(root, transform=transform)

        self.split = split
        self.tokenizer = create_tokenizer(tok_name, lowercase=True, dropout=dropout)
        self.context_length = context_length

        self.tokenizer.add_special_tokens(["[PAD]"])
        self.tokenizer.enable_padding(length=self.context_length,
                                      pad_id=self.tokenizer.token_to_id("[PAD]"))
        self.tokenizer.enable_truncation(max_length=self.context_length)

        self.items = []

        for line in open(f'{self.root}/{split}_list.txt', 'r').readlines():
            toks = line.strip().split('\t')
            assert len(toks) == 2
            (imgpath, text) = toks
            self.items.append((os.path.join(self.root, imgpath), text))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, item):
        imgpath, text = self.items[item]

        img = Image.open(imgpath).convert('RGB')
        if self.transform:
            img = self.transform(img)

        output = self.tokenizer.encode(text)
        ids = output.ids
        if not isinstance(ids, torch.LongTensor):
            ids = torch.LongTensor(ids)

        return img, ids


class Cc3mRawTextOnly(torch.utils.data.Dataset):

    def __init__(self, root, split):

        self.root = root
        self.items = []
        for line in open(f'{self.root}/{split}_list.txt', 'r').readlines():
            toks = line.strip().split('\t')
            assert len(toks) == 2
            (_, text) = toks
            self.items.append(text)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, item):
        text = self.items[item]
        return text


class Cc3mTextOnly(Cc3m):

    def __getitem__(self, item):
        _, text = self.items[item]

        output = self.tokenizer.encode(text)
        ids = output.ids
        if not isinstance(ids, torch.LongTensor):
            ids = torch.LongTensor(ids)

        return 0, ids


class Cc3mRawText(VisionDataset):
    splits = {'train', 'val'}

    def __init__(self, root, split, transform=None):
        assert split in self.splits, f'{split} is not in {self.splits}'
        super().__init__(root, transform=transform)

        self.split = split
        self.items = []

        for line in open(f'{self.root}/{split}_list.txt', 'r').readlines():
            toks = line.strip().split('\t')
            assert len(toks) == 2
            (imgpath, text) = toks
            self.items.append((os.path.join(self.root, imgpath), text))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, item):
        imgpath, text = self.items[item]

        img = Image.open(imgpath).convert('RGB')
        if self.transform:
            img = self.transform(img)

        return img, text
