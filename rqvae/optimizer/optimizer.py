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

import torch

def create_resnet_optimizer(model, config):
    optimizer_type = config.type.lower()
    if optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.init_lr, weight_decay=config.weight_decay,
            betas=config.betas
        )
    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.init_lr, weight_decay=config.weight_decay, betas=config.betas
        )
    elif optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=config.init_lr, weight_decay=config.weight_decay, momentum=0.9
        )
    else:
        raise ValueError(f'{optimizer_type} invalid..')
    return optimizer


def create_optimizer(model, config):
    arch_type = config.arch.type.lower()
    if 'rq-vae' in config.arch.type:
        optimizer = create_resnet_optimizer(model, config.optimizer)
    else:
        raise ValueError(f'{arch_type} invalid..')
    return optimizer
