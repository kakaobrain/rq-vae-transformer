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
from pathlib import Path

import torchvision


class ImageFolder(torchvision.datasets.VisionDataset):

    def __init__(self, root, train_list_file, val_list_file, split='train', **kwargs):

        root = Path(root)
        super().__init__(root, **kwargs)

        self.train_list_file = train_list_file
        self.val_list_file = val_list_file

        self.split = self._verify_split(split)

        self.loader = torchvision.datasets.folder.default_loader
        self.extensions = torchvision.datasets.folder.IMG_EXTENSIONS

        if self.split == 'trainval':
            fname_list = os.listdir(self.root)
            samples = [self.root.joinpath(fname) for fname in fname_list
                       if fname.lower().endswith(self.extensions)]
        else:
            listfile = self.train_list_file if self.split == 'train' else self.val_list_file
            with open(listfile, 'r') as f:
                samples = [self.root.joinpath(line.strip()) for line in f.readlines()]

        self.samples = samples

    def _verify_split(self, split):
        if split not in self.valid_splits:
            msg = "Unknown split {} .".format(split)
            msg += "Valid splits are {{}}.".format(", ".join(self.valid_splits))
            raise ValueError(msg)
        return split

    @property
    def valid_splits(self):
        return 'train', 'val', 'trainval'

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index, with_transform=True):
        path = self.samples[index]
        sample = self.loader(path)
        if self.transforms is not None and with_transform:
            sample, _ = self.transforms(sample, None)
        return sample, 0


class FFHQ(ImageFolder):
    train_list_file = Path(__file__).parent.joinpath('assets/ffhqtrain.txt')
    val_list_file = Path(__file__).parent.joinpath('assets/ffhqvalidation.txt')

    def __init__(self, root, split='train', **kwargs):
        super().__init__(root, FFHQ.train_list_file, FFHQ.val_list_file, split, **kwargs)
