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
import logging

import torch

from torch.utils.data.dataloader import DataLoader
from torch.cuda.amp import GradScaler


logger = logging.getLogger(__name__)
SMOKE_TEST = bool(os.environ.get("SMOKE_TEST", 0))


class TrainerTemplate():

    def __init__(self,
                 model,
                 model_ema,
                 dataset_trn,
                 dataset_val,
                 config,
                 writer,
                 device,
                 distenv,
                 model_aux=None,
                 *,
                 disc_state_dict=None,  # only used in VQGAN trainer
                 ):
        super().__init__()

        num_workers = 16

        if SMOKE_TEST:
            if not torch.distributed.is_initialized():
                num_workers = 0
            config.experiment.test_freq = 1
            config.experiment.save_ckpt_freq = 1

        self.model = model
        self.model_ema = model_ema
        self.model_aux = model_aux

        self.config = config
        self.writer = writer
        self.device = device
        self.distenv = distenv

        self.dataset_trn = dataset_trn
        self.dataset_val = dataset_val

        self.sampler_trn = torch.utils.data.distributed.DistributedSampler(
            self.dataset_trn,
            num_replicas=self.distenv.world_size,
            rank=self.distenv.world_rank,
            shuffle=True,
            seed=self.config.seed,
        )
        self.loader_trn = DataLoader(
            self.dataset_trn, sampler=self.sampler_trn, shuffle=False, pin_memory=True,
            batch_size=config.experiment.batch_size,
            num_workers=num_workers,
        )

        self.sampler_val = torch.utils.data.distributed.DistributedSampler(
            self.dataset_val,
            num_replicas=self.distenv.world_size,
            rank=self.distenv.world_rank,
            shuffle=False
        )
        self.loader_val = DataLoader(
            self.dataset_val, sampler=self.sampler_val, shuffle=False, pin_memory=True,
            batch_size=config.experiment.batch_size,
            num_workers=num_workers
        )

    def train(self, optimizer=None, scheduler=None, scaler=None, epoch=0):
        raise NotImplementedError

    def eval(self, valid=True, ema=False, verbose=False, epoch=0):
        raise NotImplementedError

    def run_epoch(self, optimizer=None, scheduler=None, epoch_st=0):
        scaler = GradScaler() if self.config.experiment.amp else None

        for i in range(epoch_st, self.config.experiment.epochs):
            self.sampler_trn.set_epoch(i)
            torch.cuda.empty_cache()
            summary_trn = self.train(optimizer, scheduler, scaler, epoch=i)
            if i == 0 or (i+1) % self.config.experiment.test_freq == 0:
                torch.cuda.empty_cache()
                summary_val = self.eval(epoch=i)
                if self.model_ema is not None:
                    summary_val_ema = self.eval(ema=True, epoch=i)

            if self.distenv.master:
                self.logging(summary_trn, scheduler=scheduler, epoch=i+1, mode='train')

                if i == 0 or (i+1) % self.config.experiment.test_freq == 0:
                    self.logging(summary_val, scheduler=scheduler, epoch=i+1, mode='valid')
                    if self.model_ema is not None:
                        self.logging(summary_val_ema, scheduler=scheduler, epoch=i+1, mode='valid_ema')

                if (i+1) % self.config.experiment.save_ckpt_freq == 0:
                    self.save_ckpt(optimizer, scheduler, i+1)

    def save_ckpt(self, optimizer, scheduler, epoch):
        ckpt_path = os.path.join(self.config.result_path, 'epoch%d_model.pt' % epoch)
        logger.info("epoch: %d, saving %s", epoch, ckpt_path)
        ckpt = {
            'epoch': epoch,
            'state_dict': self.model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        if self.model_ema is not None:
            ckpt.update(state_dict_ema=self.model_ema.module.module.state_dict())
        torch.save(ckpt, ckpt_path)
