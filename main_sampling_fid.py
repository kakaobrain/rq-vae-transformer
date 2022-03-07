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
from datetime import datetime
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torchvision
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

import rqvae.utils.dist as dist_utils
from rqvae.models import create_model
from rqvae.metrics.fid import compute_statistics_from_files
from rqvae.utils.utils import set_seed, save_pickle
from rqvae.utils.config import load_config, augment_arch_defaults

from compute_metrics import compute_metrics


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--model-ar-path', type=str, default=None)
    parser.add_argument('-v', '--model-vqvae-path', type=str, default=None)

    parser.add_argument('-n', '--n-samples', type=int, default=50000, help='total number of samples')

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

    dataset_name = config.dataset.type
    args.dataset = dataset_name


def get_logger(log_file_path=None):
    if log_file_path:
        handlers = [logging.FileHandler(log_file_path), logging.StreamHandler()]
    else:
        handlers = [logging.StreamHandler()]
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=handlers,
    )
    logger = logging.getLogger(__name__)
    return logger


def setup_logging(args, seed, distenv, device):

    seed_as_tensor = torch.tensor([seed], dtype=torch.long, device=device)
    seed_as_tensor = dist_utils.all_gather_cat(distenv, seed_as_tensor)

    if distenv.master:
        now = datetime.now().strftime('%d%m%Y_%H%M%S')
        now = now + '_' + str(seed)

        if isinstance(args.top_k, Iterable):
            top_k_str = '_'.join([str(k) for k in args.top_k])
        else:
            top_k_str = str(args.top_k)

        if isinstance(args.top_p, Iterable):
            top_p_str = '_'.join([str(p) for p in args.top_p])
        else:
            top_p_str = str(args.top_p)

        ckpt_name = os.path.basename(args.model_ar_path).split('.')[0]

        if args.save_dir:
            exp_dir_name = (Path(args.model_ar_path).parent.parent.name
                            + '_'
                            + Path(args.model_ar_path).parent.name
                            )
            save_dir = Path(args.save_dir).joinpath(exp_dir_name)
        else:
            save_dir = Path(args.model_ar_path).parent

        result_path = save_dir.joinpath('logs_fid',
                                        'valid_ema' if args.ema else 'valid',
                                        f'{ckpt_name}_temp_{args.temp:.1f}_top_k_{top_k_str}_top_p_{top_p_str}',
                                        now,
                                        )
        os.makedirs(result_path)
        writer = SummaryWriter(result_path) if args.tensorboard else None

        logger = get_logger(result_path / 'sampling.log')
        logger.info(f'[state] result_path: {result_path}')
        logger.info(f'[state] seeds: {list(seed_as_tensor.cpu().numpy())}')
        torch.save(seed_as_tensor, result_path.joinpath('seeds.pth'))
    else:
        result_path, writer = '', None
        logger = get_logger()

    return result_path, logger, writer


def load_model(path, ema=False):
    model_config = os.path.join(os.path.dirname(path), 'config.yaml')
    config = load_config(model_config)
    config.arch = augment_arch_defaults(config.arch)

    model, _ = create_model(config.arch, ema=False)
    if ema:
        ckpt = torch.load(path, map_location='cpu')['state_dict_ema']
    else:
        ckpt = torch.load(path, map_location='cpu')['state_dict']
    model.load_state_dict(ckpt)

    return model, config


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

    # load the checkpoint of rq-transformer
    model_ar, config = load_model(args.model_ar_path, ema=args.ema)
    n_labels = config.arch.vocab_size_cond  # 1 if unconditional, else the number of classes.
    
    # load the checkpoint of rq-vae
    vqvae_path = args.model_vqvae_path
    model_vqvae, _ = load_model(vqvae_path)

    model_vqvae = model_vqvae.to(device)
    model_ar = model_ar.to(device)

    model_ar = dist_utils.dataparallel_and_sync(distenv, model_ar)

    model_vqvae.eval()
    model_ar.eval()

    assert args.n_samples % n_labels == 0
    assert args.n_samples % (args.batch_size * distenv.world_size) == 0

    batch_size = args.batch_size
    num_total_batches = args.n_samples // batch_size
    num_batches = num_total_batches // distenv.world_size

    all_conds = torch.arange(0, n_labels).repeat_interleave(args.n_samples // n_labels)
    all_conds = all_conds.reshape(num_batches, distenv.world_size, batch_size)

    if distenv.master:
        logger.info(f'[state] batch_size (per gpu): {batch_size}')

    sample_shape = model_ar.module.get_block_size()

    for batch_idx in range(0, num_batches):

        class_cond = all_conds[batch_idx, distenv.world_rank].to(device)

        # Sampling quantized codes
        partial_sample = torch.zeros(batch_size, *sample_shape, dtype=torch.long, device=device)
        pixels = model_ar.module.sample(partial_sample,
                                        model_vqvae,
                                        cond=class_cond,
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
        targets = dist_utils.all_gather_cat(distenv, class_cond)

        if distenv.master:
            logger.info(f'sync pixels: {pixels.shape}')
            save_pickle(
                os.path.join(result_path, f'samples_({batch_idx+1}_{num_batches}).pkl'),
                pixels.cpu().numpy(),
            )
            # to avoid pkl files to be loaded by :meth:compute_statistics_from_files
            unique_labels = torch.unique(targets).cpu()
            logger.info(f'labels: {unique_labels}')
            np.savez(
                os.path.join(result_path, f'targets_({batch_idx+1}_{num_batches}).npz'),
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
            logger.info(f'[state] acts and stats saved at {acts_path}')

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
