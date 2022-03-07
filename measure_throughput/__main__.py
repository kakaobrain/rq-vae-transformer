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

from dataclasses import dataclass
import platform
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from rqvae.models import create_model as create_model_
from rqvae.utils.config import load_config, augment_arch_defaults

defaults = load_config(Path(__file__).parent / 'rq_defaults.yaml')


class RQVAEConfig:
    @staticmethod
    def f32(depth, codebook_size):
        # 8x8 spatial resolution
        config = defaults.rqvae_f32.copy()
        config.hparams.code_shape = [8, 8, depth]
        config.hparams.n_embed = codebook_size
        config = augment_arch_defaults(config)
        model, _ = create_model_(config)
        return model

    @staticmethod
    def f16(depth, codebook_size):
        # 16x16 spatial resolution
        config = defaults.rqvae_f16.copy()
        config.hparams.code_shape = [16, 16, depth]
        config.hparams.n_embed = codebook_size
        config = augment_arch_defaults(config)
        model, _ = create_model_(config)
        return model

    @staticmethod
    def f8(depth, codebook_size):
        # 32x32 spatial resolution
        config = defaults.rqvae_f8.copy()
        config.hparams.code_shape = [32, 32, depth]
        config.hparams.n_embed = codebook_size
        config = augment_arch_defaults(config)
        model, _ = create_model_(config)
        return model


RQVAES = {
    'f32': RQVAEConfig.f32,
    'f16': RQVAEConfig.f16,
    'f8': RQVAEConfig.f8,
}


class RQTransformerConfig:
    @staticmethod
    def huge(code_shape, codebook_size):
        # 1400M config
        config = defaults.rqtransformer.copy()

        config.block_size = code_shape
        config.vocab_size = codebook_size

        config.embed_dim = 1536

        if code_shape[-1] > 1:
            config.body.n_layer = 42
            config.head.n_layer = 6
        else:
            config.body.n_layer = 48
            config.head.n_layer = 0

        config.body.block.n_head = 24
        config.head.block.n_head = 24

        config = augment_arch_defaults(config)
        model, _ = create_model_(config)
        return model

    @staticmethod
    def large(code_shape, codebook_size):
        # 800M config
        config = defaults.rqtransformer.copy()

        config.block_size = code_shape
        config.vocab_size = codebook_size

        config.embed_dim = 1536

        if code_shape[-1] > 1:
            config.body.n_layer = 24
            config.head.n_layer = 4
        else:
            config.body.n_layer = 28
            config.head.n_layer = 0

        config.body.block.n_head = 24
        config.head.block.n_head = 24

        config = augment_arch_defaults(config)
        model, _ = create_model_(config)
        return model

    @staticmethod
    def medium(code_shape, codebook_size):
        # 350M config
        config = defaults.rqtransformer.copy()

        config.block_size = code_shape
        config.vocab_size = codebook_size

        config.embed_dim = 1024

        if code_shape[-1] > 1:
            config.body.n_layer = 24
            config.head.n_layer = 4
        else:
            config.body.n_layer = 28
            config.head.n_layer = 0

        config.body.block.n_head = 16
        config.head.block.n_head = 16

        config = augment_arch_defaults(config)
        model, _ = create_model_(config)
        return model

    @staticmethod
    def small(code_shape, codebook_size):
        # 90M config
        config = defaults.rqtransformer.copy()

        config.block_size = code_shape
        config.vocab_size = codebook_size

        config.embed_dim = 512

        if code_shape[-1] > 1:
            config.body.n_layer = 24
            config.head.n_layer = 4
        else:
            config.body.n_layer = 28
            config.head.n_layer = 0

        config.body.block.n_head = 8
        config.head.block.n_head = 8

        config = augment_arch_defaults(config)
        model, _ = create_model_(config)
        return model

    @staticmethod
    def vqgan_large(code_shape, codebook_size):
        # 800M config
        config = defaults.rqtransformer.copy()

        if tuple(code_shape) != (16, 16, 1) or codebook_size != 1024:
            raise ValueError("vqgan_large only works with f16-d1-c1024")

        config.block_size = code_shape
        config.vocab_size = codebook_size

        config.embed_dim = 1664

        config.body.n_layer = 24
        config.head.n_layer = 0

        config.body.block.n_head = 16
        config.head.block.n_head = 16

        config = augment_arch_defaults(config)
        model, _ = create_model_(config)
        return model

    @staticmethod
    def vqgan_huge(code_shape, codebook_size):
        # 1400M config
        config = defaults.rqtransformer.copy()

        if tuple(code_shape) != (16, 16, 1) or codebook_size != 16384:
            raise ValueError("vqgan_huge only works with f16-d1-c16384")

        config.block_size = code_shape
        config.vocab_size = codebook_size

        config.embed_dim = 1536

        config.body.n_layer = 48
        config.head.n_layer = 0

        config.body.block.n_head = 24
        config.head.block.n_head = 24

        config = augment_arch_defaults(config)
        model, _ = create_model_(config)
        return model


RQTRANSFORMERS = {
    'huge': RQTransformerConfig.huge,
    'large': RQTransformerConfig.large,
    'medium': RQTransformerConfig.medium,
    'small': RQTransformerConfig.small,
    'vqgan_large': RQTransformerConfig.vqgan_large,
    'vqgan_huge': RQTransformerConfig.vqgan_huge,
}


def create_model(rqvae, rqtransformer, depth, codebook_size):

    rqvae_model = RQVAES[rqvae](depth, codebook_size)
    code_shape = rqvae_model.code_shape

    rqtransformer_model = RQTRANSFORMERS[rqtransformer](code_shape, codebook_size)

    return rqvae_model, rqtransformer_model


@dataclass
class Experiment:
    f: int = 32
    model: str = 'huge'
    d: int = 4
    c: int = 16384

    batch_size: int = 50

    n_loop: int = 6
    warmup: int = 1


def main(args: Experiment):
    torch.set_grad_enabled(False)

    rqvae = f'f{args.f}'
    rqtransformer = args.model
    depth = args.d
    codebook_size = args.c

    model_aux, model_ar = create_model(rqvae, rqtransformer, depth, codebook_size)

    device = torch.device('cuda')
    model_aux, model_ar = model_aux.to(device), model_ar.to(device)
    model_aux.eval()
    model_ar.eval()

    title = (f'{rqvae}-{rqtransformer}-d{depth}-c{codebook_size}-bs{args.batch_size}, '
             f'sampling loops {args.warmup+1}-{args.n_loop}'
             )
    print(title)
    print('python: %s, torch: %s, cudnn: %s, cuda: %s, gpu: %s' % (
        platform.python_version(),
        torch.__version__,
        torch.backends.cudnn.version(),
        torch.version.cuda,
        torch.cuda.get_device_name(device)
    ))

    aux_size = sum([p.numel() for p in model_aux.parameters()]) / (10 ** 6)
    ar_size = sum([p.numel() for p in model_ar.parameters()]) / (10 ** 6)
    print(f'rqgan size: {aux_size:.1f}M, rqtransformer size: {ar_size:.1f}M')

    batch_size = args.batch_size
    n_iter_per_loop = (1000 + batch_size - 1) // batch_size
    n_loop = args.n_loop

    empty_sample = torch.zeros(batch_size, *model_ar.block_size, device=device, dtype=torch.long)
    empty_cond = torch.zeros(batch_size, model_ar.block_size_cond, device=device, dtype=torch.long)

    def loop(loop_idx: int):
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter_per_loop)]
        middles = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter_per_loop)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter_per_loop)]

        torch.cuda.synchronize(device)
        tic = time.time()

        pbar = tqdm(range(n_iter_per_loop), total=n_iter_per_loop)
        for i in pbar:
            starts[i].record()
            codes = model_ar.sample(empty_sample, model_aux=model_aux, cond=empty_cond)

            middles[i].record()

            chunks = codes.chunk(batch_size)
            pixels = torch.cat([model_aux.decode_code(chunk) for chunk in chunks], dim=0)
            _ = (0.5 * pixels + 0.5).clamp(0, 1)
            ends[i].record()

            speed = (time.time() - tic) / ((i + 1) * batch_size) * 1000
            pbar.set_description(f'{loop_idx+1}/{n_loop} | {speed:.3f} ms/sample (estimated)')

        torch.cuda.synchronize(device)
        toc = time.time()

        elapsed_time = toc - tic
        elapsed_time_ar = sum([starts[i].elapsed_time(middles[i]) for i in range(n_iter_per_loop)]) / 1000
        elapsed_time_decode = sum([middles[i].elapsed_time(ends[i]) for i in range(n_iter_per_loop)]) / 1000
        print(f'{loop_idx + 1}/{n_loop} | '
              f'{elapsed_time:.1f} s/loop (ar: {elapsed_time_ar:.1f}, decode: {elapsed_time_decode:.1f})')

        speed = elapsed_time / (n_iter_per_loop * batch_size) * 1000
        speed_ar = elapsed_time_ar / (n_iter_per_loop * batch_size) * 1000
        speed_decode = elapsed_time_decode / (n_iter_per_loop * batch_size) * 1000
        print(f'{loop_idx+1}/{n_loop} | {speed:.1f} ms/sample (ar: {speed_ar:.1f}, decode: {speed_decode:.1f})')

        return speed, speed_ar, speed_decode

    speeds = []
    speeds_ar = []
    speeds_decode = []

    print('-' * 80)
    for loop_idx in range(args.n_loop):
        speed, speed_ar, speed_decode = loop(loop_idx)

        if loop_idx < args.warmup:
            continue

        speeds.append(speed)
        speeds_ar.append(speed_ar)
        speeds_decode.append(speed_decode)
    print('-' * 80)

    n = len(speeds)
    speed = sum(speeds) / n
    speed_ar = sum(speeds_ar) / n
    speed_decode = sum(speeds_decode) / n
    print(f'{title} | {speed:.4f} ms/sample (ar: {speed_ar:.4f}, decode: {speed_decode:.4f})')
    print('=' * 80)


if __name__ == '__main__':

    args = OmegaConf.merge(OmegaConf.structured(Experiment()), OmegaConf.from_cli())
    main(args)
