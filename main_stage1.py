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
import argparse
import math

import torch
import torch.distributed as dist

import rqvae.utils.dist as dist_utils
from rqvae.models import create_model
from rqvae.trainers import create_trainer
from rqvae.img_datasets import create_dataset
from rqvae.optimizer import create_optimizer, create_scheduler
from rqvae.utils.utils import set_seed, compute_model_size, get_num_conv_linear_layers
from rqvae.utils.setup import setup


parser = argparse.ArgumentParser()

parser.add_argument('-m', '--model-config', type=str, default='./configs/c10-igpt.yaml')
parser.add_argument('-r', '--result-path', type=str, default='./results.tmp')
parser.add_argument('-l', '--load-path', type=str, default='')
parser.add_argument('-t', '--test-batch-size', type=int, default=200)
parser.add_argument('-e', '--test-epoch', type=int, default=-1)
parser.add_argument('-p', '--postfix', type=str, default='')
parser.add_argument('--seed', type=int, default=0)

parser.add_argument('--world_size', default=-1, type=int, help='number of nodes for distributed training')
parser.add_argument('--local_rank', default=-1, type=int, help='local rank for distributed training')
parser.add_argument('--node_rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--timeout', type=int, default=86400, help='time limit (s) to wait for other nodes in DDP')
parser.add_argument('--eval', action='store_true')
parser.add_argument('--resume', action='store_true')

args, extra_args = parser.parse_known_args()

set_seed(args.seed)


if __name__ == '__main__':

    config, logger, writer = setup(args, extra_args)
    distenv = config.runtime.distenv

    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda', distenv.local_rank)
    torch.cuda.set_device(device)

    dataset_trn, dataset_val = create_dataset(config, is_eval=args.eval, logger=logger)
    model, model_ema = create_model(config.arch, ema=config.arch.ema is not None)
    model = model.to(device)
    if model_ema:
        model_ema = model_ema.to(device)
    trainer = create_trainer(config)

    train_epochs = config.experiment.epochs
    steps_per_epoch = math.ceil(len(dataset_trn) / (config.experiment.batch_size * distenv.world_size))
    epoch_st = 0

    if distenv.master:
        logger.info(f'#conv+linear layers: {get_num_conv_linear_layers(model)}')

    if not args.eval:
        optimizer = create_optimizer(model, config)
        scheduler = create_scheduler(
            optimizer, config.optimizer.warmup, steps_per_epoch,
            config.experiment.epochs, distenv
        )

    disc_state_dict = None
    if not args.load_path == '':
        ckpt = torch.load(args.load_path, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'])
        disc_state_dict = ckpt.get('discriminator', None)
        if model_ema:
            model_ema.load_state_dict(ckpt['state_dict_ema'])
        
        if args.resume:
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
            epoch_st = ckpt['epoch']
            
        if distenv.master:
            logger.info(f'{args.load_path} model is loaded')
            if args.resume:
                logger.info(f'Optimizer, scheduelr, and epoch is resumed')

    if distenv.master:
        print(model)
        compute_model_size(model, logger)

    if distenv.master and not args.eval:
        logger.info(optimizer.__repr__())

    model = dist_utils.dataparallel_and_sync(distenv, model)
    if model_ema:
        model_ema = dist_utils.dataparallel_and_sync(distenv, model_ema)
    trainer = trainer(model, model_ema, dataset_trn, dataset_val, config, writer,
                      device, distenv, disc_state_dict=disc_state_dict)
    if args.eval:
        trainer.eval(valid=False, verbose=True)
        trainer.eval(valid=True, verbose=True)
        if model_ema:
            trainer.eval(valid=True, ema=True, verbose=True)
    else:
        trainer.run_epoch(optimizer, scheduler, epoch_st)

    dist.barrier()

    if distenv.master:
        writer.close()  # may prevent from a file stable error in brain cloud..
