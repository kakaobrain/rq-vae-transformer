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
from pathlib import Path

import numpy as np
import torch
import torchvision
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

import rqvae.utils.dist as dist_utils
from rqvae.txtimg_datasets.cc3m import Cc3mTextOnly
from rqvae.txtimg_datasets.coco import CocoTextOnly
from rqvae.metrics.fid import compute_statistics_from_files
from rqvae.utils.utils import set_seed, save_pickle
from rqvae.utils.config import load_config

from main_sampling_fid import (setup_logging,
                               load_model,
                               compute_metrics)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--model-ar-path', type=str, default=None)
    parser.add_argument('-v', '--model-vqvae-path', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='cc3m', choices=['cc3m', 'coco_2014val'])

    parser.add_argument('-t', '--temp', type=float, default=None)
    parser.add_argument('--top-k', type=int, nargs='+', default=None)
    parser.add_argument('--top-p', type=float, nargs='+', default=None)
    parser.add_argument('-bs', '--batch-size', type=int, default=100, help='batch size (per gpu)')

    parser.add_argument('--ema', action='store_true')
    parser.add_argument('--save-dir', type=str, default=None)
    parser.add_argument('--no-tensorboard', action='store_false', dest='tensorboard')
    parser.add_argument('--no-stats-saving', action='store_false', dest='stats_saving')
    parser.add_argument('--seed', type=int, default=None)

    parser.add_argument('--world_size', default=-1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--timeout', type=int, default=86400, help='time limit (s) to wait for other nodes in DDP')

    return parser


def add_default_args(args):

    config_path = Path(args.model_ar_path).parent / 'config.yaml'
    config = load_config(config_path)

    if args.temp is None:
        args.temp = config.sampling.temp

    if args.top_k is None:
        args.top_k = config.sampling.top_k

    if args.top_p is None:
        args.top_p = config.sampling.top_p


def get_text_loader(args, config, distenv):
    valid_transform = [
        torchvision.transforms.Resize(size=(config.dataset.image_resolution, config.dataset.image_resolution)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    if args.dataset == 'cc3m':
        root = config.dataset.get('root', 'data/cc3m')
        dataset_val = Cc3mTextOnly(
            root,
            split='val',
            tok_name=config.dataset.txt_tok_name,
            transform=valid_transform,
            context_length=config.dataset.context_length,
            dropout=None,
        )
    elif args.dataset == 'coco_2014val':
        root = config.dataset.get('root', 'data/coco')
        dataset_val = CocoTextOnly( 
            root,
            split='val',
            tok_name=config.dataset.txt_tok_name,
            transform=valid_transform,
            context_length=config.dataset.context_length,
            dropout=None,
        )
    else:
        raise NotImplementedError
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset_val,
        num_replicas=distenv.world_size,
        rank=distenv.world_rank,
        shuffle=False
    )
    loader = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=0
    )
    return loader


def main(args):
    torch.backends.cudnn.benchmark = True
    distenv = dist_utils.initialize(args)
    device = torch.device('cuda', distenv.local_rank)
    torch.cuda.set_device(device)

    if args.seed:
        seed = set_seed(args.seed + distenv.world_rank)
    else:
        seed = set_seed(None)

    result_path, logger, writer = setup_logging(args, seed, distenv, device)

    # load the checkpoint of RQ-Transformer
    model_ar, config = load_model(args.model_ar_path, ema=args.ema)

    # load the checkpoint of RQ-VAE
    vqvae_path = args.model_vqvae_path
    model_vqvae, _ = load_model(vqvae_path)

    model_vqvae = model_vqvae.to(device)
    model_ar = model_ar.to(device)

    model_ar = dist_utils.dataparallel_and_sync(distenv, model_ar)

    model_vqvae.eval()
    model_ar.eval()

    loader = get_text_loader(args, config, distenv)
    batch_size = args.batch_size
    num_batches = len(loader)
    if distenv.master:
        logger.info(f'[state] batch_size (per gpu): {batch_size}')
        logger.info(f'[state] n_batches: {len(loader)}x{batch_size*distenv.world_size}'
                    f'={len(loader) * batch_size* distenv.world_size}')

    sample_shape = model_ar.module.get_block_size()

    def get_initial_sample(n_samples):
        return torch.zeros(n_samples, *sample_shape, dtype=torch.long, device=device)

    for batch_idx, (_, txts) in enumerate(loader):

        # Sampling quantized codes
        txts = txts.to(device)
        partial_sample = get_initial_sample(txts.shape[0])
        pixels = model_ar.module.sample(partial_sample,
                                        model_vqvae,
                                        cond=txts,
                                        temperature=args.temp,
                                        top_k=args.top_k,
                                        top_p=args.top_p,
                                        amp=True,
                                        fast=True,
                                        is_tqdm=distenv.master,
                                        desc=f"(sampling {batch_idx+1}/{num_batches})",
                                        )

        # Decoding the sampled codes into RGB images
        pixels = torch.cat([model_vqvae.decode_code(pixels[i:i+1]) for i in range(pixels.size(0))], dim=0)
        pixels = pixels * 0.5 + 0.5
        pixels = torch.clamp(pixels, 0, 1)
        pixels = dist_utils.all_gather_cat(distenv, pixels)
        targets = dist_utils.all_gather_cat(distenv, txts)

        if distenv.master:
            # (M * B) -> (M, B) -> (B, M) -> (B * M)
            # to retain sample order same as in the dataset
            pixels = pixels.reshape(distenv.world_size, -1, *pixels.shape[1:])
            pixels = pixels.transpose(0, 1)
            pixels = pixels.reshape(-1, *pixels.shape[2:])

            logger.info(f'sync pixels: {pixels.shape}')
            save_pickle(
                os.path.join(result_path, f'samples_({batch_idx+1}_{num_batches}).pkl'),
                pixels.cpu().numpy(),
            )

            targets = targets.reshape(distenv.world_size, -1, *targets.shape[1:])
            targets = targets.transpose(0, 1)
            targets = targets.reshape(-1, *targets.shape[2:])
            np.savez(
                os.path.join(result_path, f'targets_({batch_idx + 1}_{num_batches}).npz'),
                targets=targets.cpu().numpy(),
            )

            if writer:
                grid = torchvision.utils.make_grid(pixels[:100], nrow=10)
                writer.add_image('samples', grid, batch_idx)

        if os.environ.get("SMOKE_TEST", 0):
            break

    logger.info(f'[state] end of sampling.')
    if dist.is_initialized():
        dist.barrier()

    if distenv.master:

        # compute and save stats
        if args.stats_saving:
            mu_gen, sigma_gen, acts = compute_statistics_from_files(result_path, device=device, return_acts=True)
            acts_path = Path(result_path).joinpath('acts.npz')
            np.savez(acts_path, acts=acts, mu=mu_gen, sigma=sigma_gen)
            logger.info(f'[state] stat saved at {acts_path}')

            metrics = compute_metrics(result_path, args.dataset)
            metrics_repr = ', '.join([f'{key}: {value}' for key, value in metrics.items()])
            logger.info(f'metrics: {metrics_repr}')

        # close the tb writer
        if writer:
            writer.close()


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    add_default_args(args)
    print(args)
    if (not args.model_ar_path or not args.model_vqvae_path):
        raise Exception("Both ar_path and vqvae_path are needed for sampling")
    main(args)
