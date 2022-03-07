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
import torch.utils.data

from .transforms import create_transforms
from .coco import Coco
from .cc3m import Cc3m


def create_datasets(config, is_eval=False, logger=None):

    data_config = config.dataset

    train_transform = create_transforms(data_config, split='train', is_eval=is_eval)
    valid_transform = create_transforms(data_config, split='valid', is_eval=is_eval)

    root = data_config.get('root', None)

    if data_config.dataset == 'coco':
        root = root if root else 'data/coco'
        train_ds_cls = Coco
        valid_ds_cls = Coco
    elif data_config.dataset == 'cc3m':
        root = root if root else 'data/cc3m'
        train_ds_cls = Cc3m
        valid_ds_cls = Cc3m
    else:
        raise NotImplementedError(data_config.dataset)

    train_dataset = train_ds_cls(root,
                                 split='train',
                                 tok_name=data_config.txt_tok_name,
                                 transform=train_transform,
                                 context_length=data_config.context_length,
                                 dropout=data_config.bpe_dropout)
    valid_dataset = valid_ds_cls(root,
                                 split='val',
                                 tok_name=data_config.txt_tok_name,
                                 transform=valid_transform,
                                 context_length=data_config.context_length,
                                 dropout=None)

    if bool(os.environ.get("SMOKE_TEST", 0)):
        dataset_len = config.experiment.total_batch_size * 2
        train_dataset = torch.utils.data.Subset(train_dataset, torch.randperm(len(train_dataset))[:dataset_len])
        valid_dataset = torch.utils.data.Subset(valid_dataset, torch.randperm(len(valid_dataset))[:dataset_len])

    if logger is not None:
        logger.info(f'#train samples: {len(train_dataset)}, #valid samples: {len(valid_dataset)}')

    return train_dataset, valid_dataset
