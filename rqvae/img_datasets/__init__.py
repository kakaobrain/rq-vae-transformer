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
from torch.utils.data import Subset
import torchvision
from torchvision.datasets import ImageNet

from .lsun import LSUNClass
from .ffhq import FFHQ
from .transforms import create_transforms

SMOKE_TEST = bool(os.environ.get("SMOKE_TEST", 0))


def create_dataset(config, is_eval=False, logger=None):
    transforms_trn = create_transforms(config.dataset, split='train', is_eval=is_eval)
    transforms_val = create_transforms(config.dataset, split='val', is_eval=is_eval)

    root = config.dataset.get('root', None)

    if config.dataset.type == 'imagenet':
        root = root if root else 'data/imagenet'
        dataset_trn = ImageNet(root, split='train', transform=transforms_trn)
        dataset_val = ImageNet(root, split='val', transform=transforms_val)
    elif config.dataset.type == 'imagenet_u':
        root = root if root else 'data/imagenet'

        def target_transform(_):
            return 0
        dataset_trn = ImageNet(root, split='train', transform=transforms_trn, target_transform=target_transform)
        dataset_val = ImageNet(root, split='val', transform=transforms_val, target_transform=target_transform)
    elif config.dataset.type == 'ffhq':
        root = root if root else 'data/ffhq'
        dataset_trn = FFHQ(root, split='train', transform=transforms_trn)
        dataset_val = FFHQ(root, split='val', transform=transforms_val)
    elif config.dataset.type in ['LSUN-cat', 'LSUN-church', 'LSUN-bedroom']:
        root = root if root else 'data/lsun'
        category_name = config.dataset.type.split('-')[-1]
        dataset_trn = LSUNClass(root, category_name=category_name, transform=transforms_trn)
        dataset_val = LSUNClass(root, category_name=category_name, transform=transforms_trn)
    else:
        raise ValueError('%s not supported...' % config.dataset.type)

    if SMOKE_TEST:
        dataset_len = config.experiment.total_batch_size * 2
        dataset_trn = torch.utils.data.Subset(dataset_trn, torch.randperm(len(dataset_trn))[:dataset_len])
        dataset_val = torch.utils.data.Subset(dataset_val, torch.randperm(len(dataset_val))[:dataset_len])

    if logger is not None:
        logger.info(f'#train samples: {len(dataset_trn)}, #valid samples: {len(dataset_val)}')

    return dataset_trn, dataset_val
