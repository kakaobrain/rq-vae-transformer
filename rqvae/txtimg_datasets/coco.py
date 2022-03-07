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

import random

import torch

from torchvision.datasets import CocoCaptions, VisionDataset

from .tokenizers import create_tokenizer


class Coco(VisionDataset):
    splits = {'val'}

    def __init__(self, root, split, tok_name, transform=None, context_length=77, dropout=None):
        assert split in self.splits, f'{split} is not in {self.splits}'
        super().__init__(root, transform=transform)

        self.split = split
        self.tokenizer = create_tokenizer(tok_name, lowercase=True, dropout=dropout)
        self.context_length = context_length

        self.dataset = CocoCaptions(root=f'{self.root}/images/val2014',
                                    annFile=f'{self.root}/annotations/captions_val2014_30K_samples.json')

        self.tokenizer.add_special_tokens(["[PAD]"])
        self.tokenizer.enable_padding(length=self.context_length,
                                      pad_id=self.tokenizer.token_to_id("[PAD]"))
        self.tokenizer.enable_truncation(max_length=self.context_length)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        img, text = self.dataset[item]

        if self.transform:
            img = self.transform(img)

        # text = ' '.join(text)  # text is a list of sentences. Concat them.
        if self.split == 'train':
            rnd_txt = random.randint(0, len(text)-1)
            text = text[rnd_txt]
        else:
            text = text[0]

        output = self.tokenizer.encode(text)
        ids = output.ids
        if not isinstance(ids, torch.LongTensor):
            ids = torch.LongTensor(ids)

        return img, ids


class CocoTextOnly(Coco):

    def __getitem__(self, item):
        _, text = self.dataset[item]

        text = text[0]

        output = self.tokenizer.encode(text)
        ids = output.ids
        if not isinstance(ids, torch.LongTensor):
            ids = torch.LongTensor(ids)

        return 0, ids


class CocoRawText(VisionDataset):
    splits = {'val'}

    def __init__(self, root, split, transform=None):
        assert split in self.splits, f'{split} is not in {self.splits}'
        super().__init__(root, transform=transform)

        self.split = split

        self.dataset = CocoCaptions(root=f'{self.root}/images/val2014',
                                    annFile=f'{self.root}/annotations/captions_val2014_30K_samples.json')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        img, text = self.dataset[item]

        if self.transform:
            img = self.transform(img)

        text = text[0]

        return img, text


class CocoRawTextOnly(CocoRawText):
    def __getitem__(self, item):
        _, text = self.dataset[item]
        return text
