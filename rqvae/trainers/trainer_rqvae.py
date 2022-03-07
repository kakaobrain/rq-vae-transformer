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

import logging
import os

import torch
import torchvision
from tqdm import tqdm

from rqvae.losses.vqgan import create_vqgan_loss, create_discriminator_with_optimizer_scheduler
import rqvae.utils.dist as dist_utils

from .accumulator import AccmStage1WithGAN
from .trainer import TrainerTemplate

logger = logging.getLogger(__name__)


def calculate_adaptive_weight(nll_loss, g_loss, last_layer):
    nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
    g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

    d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
    d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
    return d_weight


class Trainer(TrainerTemplate):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert len(self.config.arch.hparams.code_shape) in [2, 3]

        if len(self.config.arch.hparams.code_shape) == 2:
            self.n_codebook = 1
        else:
            self.n_codebook = self.config.arch.hparams.code_shape[-1]

        # GAN related part
        gan_config = self.config.gan

        disc_config = gan_config.disc
        self.gan_start_epoch = gan_config.loss.disc_start
        num_epochs_for_gan = self.config.experiment.epochs - self.gan_start_epoch

        disc_model, disc_optim, disc_sched = \
            create_discriminator_with_optimizer_scheduler(disc_config,
                                                          steps_per_epoch=len(self.loader_trn),
                                                          max_epoch=num_epochs_for_gan,
                                                          distenv=self.distenv,
                                                          )
        disc_state_dict = kwargs.get('disc_state_dict', None)
        if disc_state_dict is not None:
            disc_model.load_state_dict(disc_state_dict)
            logger.info('[state] discriminator loaded')
        disc_model = disc_model.to(self.device)

        self.discriminator = dist_utils.dataparallel_and_sync(self.distenv, disc_model)
        self.disc_optimizer = disc_optim
        self.disc_scheduler = disc_sched

        d_loss, g_loss, p_loss = create_vqgan_loss(gan_config.loss)

        self.disc_loss = d_loss
        self.gen_loss = g_loss
        self.perceptual_loss = p_loss.to(self.device).eval()
        self.perceptual_weight = gan_config.loss.perceptual_weight
        self.disc_weight = gan_config.loss.disc_weight

        if hasattr(self.model, 'module'):
            self.get_last_layer = self.model.module.get_last_layer
        else:
            self.get_last_layer = self.model.get_last_layer

    def get_accm(self):
        config = self.config

        metric_names = [
            'loss_total', 'loss_recon', 'loss_latent',
            'loss_pcpt', 'loss_gen', 'loss_disc', 'g_weight',
            'logits_real', 'logits_fake',
        ]
        accm = AccmStage1WithGAN(
            metric_names,
            n_codebook=self.n_codebook,
            codebook_size=config.arch.hparams.n_embed,
            code_hier=self.config.arch.code_hier,
            use_padding_idx=self.config.arch.hparams.use_padding_idx,
            device=self.device,
        )

        return accm

    def gan_loss(self, inputs, recons, mode='idle'):

        loss_gen = torch.zeros((), device=self.device)
        loss_disc = torch.zeros((), device=self.device)

        logits_avg = {}

        if mode == 'gen':
            logits_fake, _ = self.discriminator(recons.contiguous(), None)
            loss_gen = self.gen_loss(logits_fake)

        elif mode == 'disc':
            logits_fake, logits_real = self.discriminator(recons.contiguous().detach(), inputs.contiguous().detach())

            loss_disc = self.disc_loss(logits_real, logits_fake)

            logits_avg['logits_real'] = logits_real.detach().mean()
            logits_avg['logits_fake'] = logits_fake.detach().mean()

        elif mode == 'eval':
            logits_fake, logits_real = self.discriminator(recons.contiguous().detach(), inputs.contiguous().detach())

            loss_gen = self.gen_loss(logits_fake)
            loss_disc = self.disc_loss(logits_real, logits_fake)

            logits_avg['logits_real'] = logits_real.detach().mean()
            logits_avg['logits_fake'] = logits_fake.detach().mean()

        return loss_gen, loss_disc, logits_avg

    @torch.no_grad()
    def eval(self, valid=True, ema=False, verbose=False, epoch=0):
        model = self.model_ema if ema else self.model
        discriminator = self.discriminator
        loader = self.loader_val if valid else self.loader_trn
        n_inst = len(self.dataset_val) if valid else len(self.dataset_trn)

        use_discriminator = True if epoch >= self.gan_start_epoch else False

        accm = self.get_accm()

        if self.distenv.master:
            pbar = tqdm(enumerate(loader), total=len(loader))
        else:
            pbar = enumerate(loader)

        model.eval()
        discriminator.eval()
        for it, inputs in pbar:
            model.zero_grad()
            xs = inputs[0].to(self.device)

            outputs = model(xs)
            xs_recon = outputs[0]
            outputs = model.module.compute_loss(*outputs, xs=xs, valid=True)

            loss_rec_lat = outputs['loss_total']
            loss_recon = outputs['loss_recon']
            loss_latent = outputs['loss_latent']

            loss_pcpt = self.perceptual_loss(xs, xs_recon)
            p_weight = self.perceptual_weight

            if use_discriminator:
                loss_gen, loss_disc, logits = self.gan_loss(xs, xs_recon, mode='eval')
            else:
                loss_gen = torch.zeros((), device=self.device)
                loss_disc = torch.zeros((), device=self.device)
                logits = {}

            loss_pcpt *= xs.size(0)  # need to be scaled with batch size
            loss_gen *= xs.size(0)
            loss_disc *= xs.size(0)
            logits = {k: v * xs.size(0) for k, v in logits.items()}

            # logging
            codes = outputs['codes']
            loss_total = loss_rec_lat + p_weight * loss_pcpt  # rec + lat + pcpt
            metrics = dict(loss_total=loss_total,
                           loss_recon=loss_recon,
                           loss_latent=loss_latent,
                           loss_pcpt=loss_pcpt,
                           loss_gen=loss_gen,
                           loss_disc=loss_disc,
                           **logits,
                           )
            accm.update(codes,
                        metrics,
                        count=xs.shape[0],
                        sync=True,
                        distenv=self.distenv)

            if self.distenv.master:
                line = accm.get_summary().print_line()
                pbar.set_description(line)

        line = accm.get_summary(n_inst).print_line()

        if self.distenv.master and verbose:
            mode = "valid" if valid else "train"
            mode = "%s_ema" % mode if ema else mode
            logger.info(f"""{mode:10s}, """ + line)
            self.reconstruct(xs, epoch=0, mode=mode)
            if self.n_codebook > 1:
                for code_idx in range(self.n_codebook):
                    self.reconstruct_partial_codes(xs, 0, code_idx, mode, 'select')
                    self.reconstruct_partial_codes(xs, 0, code_idx, mode, 'add')

        summary = accm.get_summary(n_inst)
        summary['xs'] = xs

        return summary

    def train(self, optimizer=None, scheduler=None, scaler=None, epoch=0):
        model = self.model
        model.train()

        discriminator = self.discriminator
        discriminator.train()
        use_discriminator = True if epoch >= self.gan_start_epoch else False

        accm = self.get_accm()

        if self.distenv.master:
            pbar = tqdm(enumerate(self.loader_trn), total=len(self.loader_trn))
        else:
            pbar = enumerate(self.loader_trn)

        for it, inputs in pbar:
            model.zero_grad(set_to_none=True)
            xs = inputs[0].to(self.device, non_blocking=True)

            outputs = model(xs)
            xs_recon = outputs[0]
            outputs = model.module.compute_loss(*outputs, xs=xs)

            loss_rec_lat = outputs['loss_total']
            loss_recon = outputs['loss_recon']
            loss_latent = outputs['loss_latent']

            # generator loss
            loss_pcpt = self.perceptual_loss(xs, xs_recon)
            p_weight = self.perceptual_weight

            if use_discriminator:
                loss_gen, _, _ = self.gan_loss(xs, xs_recon, mode='gen')
                g_weight = calculate_adaptive_weight(loss_recon + p_weight * loss_pcpt,
                                                     loss_gen,
                                                     last_layer=self.get_last_layer())
            else:
                loss_gen = torch.zeros((), device=self.device)
                g_weight = torch.zeros((), device=self.device)

            loss_gen_total = loss_rec_lat + p_weight * loss_pcpt + g_weight * self.disc_weight * loss_gen
            loss_gen_total.backward()

            optimizer.step()
            scheduler.step()

            # discriminator loss
            discriminator.zero_grad(set_to_none=True)

            if use_discriminator:
                _, loss_disc, logits = self.gan_loss(xs, xs_recon, mode='disc')
                (self.disc_weight * loss_disc).backward()
                self.disc_optimizer.step()
                self.disc_scheduler.step()
            else:
                loss_disc = torch.zeros((), device=self.device)
                logits = {}

            # logging
            codes = outputs['codes']
            loss_total = loss_rec_lat.detach() + p_weight * loss_pcpt.detach()  # rec + lat + pcpt
            metrics = {
                'loss_total': loss_total,
                'loss_recon': loss_recon.detach(),
                'loss_latent': loss_latent.detach(),
                'loss_pcpt': loss_pcpt.detach(),
                'loss_gen': loss_gen.detach(),
                'loss_disc': loss_disc.detach(),
                'g_weight': g_weight.detach(),
                **logits,
            }
            accm.update(codes, metrics, count=1)

            if self.distenv.master:
                line = f"""(epoch {epoch} / iter {it}) """
                line += accm.get_summary().print_line()
                line += f""", lr: {scheduler.get_last_lr()[0]:e}"""
                pbar.set_description(line)

                # per-step logging
                global_iter = epoch * len(self.loader_trn) + it
                if (global_iter+1) % 50 == 0:
                    for key, value in metrics.items():
                        self.writer.add_scalar(f'loss_step/{key}', value, 'train', global_iter)
                    self.writer.add_scalar('lr_step', scheduler.get_last_lr()[0], 'train', global_iter)
                    if use_discriminator:
                        self.writer.add_scalar('d_lr_step', self.disc_scheduler.get_last_lr()[0], 'train', global_iter)

                if (global_iter+1) % 250 == 0:
                    xs_real, xs_recon = model.module.get_recon_imgs(xs[:16], xs_recon[:16])
                    grid = torch.cat([xs_real[:8], xs_recon[:8], xs_real[8:], xs_recon[8:]], dim=0)
                    grid = torchvision.utils.make_grid(grid, nrow=8)
                    self.writer.add_image('reconstruction_step', grid, 'train', global_iter)

        summary = accm.get_summary()
        summary['xs'] = xs

        return summary

    def logging(self, summary, scheduler=None, epoch=0, mode='train'):
        if epoch % 10 == 1 or epoch % self.config.experiment.test_freq == 0:
            self.reconstruct(summary['xs'], epoch, mode)
            if self.n_codebook > 1:
                for code_idx in range(self.n_codebook):
                    self.reconstruct_partial_codes(summary['xs'], epoch, code_idx, mode, 'select')
                    self.reconstruct_partial_codes(summary['xs'], epoch, code_idx, mode, 'add')

        for key, value in summary.metrics.items():
            self.writer.add_scalar(f'loss/{key}', summary[key], mode, epoch)

        for level, ent_codes in enumerate(summary['ent_codes_wo_pad']):
            for book_idx, ent_code in enumerate(ent_codes):
                self.writer.add_scalar(f'codebooks-wo-pad/entropy-level-{level}/codebook{book_idx}',
                                       ent_code, mode, epoch)

        if summary['ent_codes_w_pad'] is not None:
            for level, ent_codes in enumerate(summary['ent_codes_w_pad']):
                for book_idx, ent_code in enumerate(ent_codes):
                    self.writer.add_scalar(f'codebooks-w-pad/entropy-level-{level}/codebook{book_idx}',
                                           ent_code, mode, epoch)

        if mode == 'train':
            self.writer.add_scalar('lr', scheduler.get_last_lr()[0], mode, epoch)

        line = f"""ep:{epoch}, {mode:10s}, """
        line += summary.print_line()
        line += f""", """
        if scheduler:
            line += f"""lr: {scheduler.get_last_lr()[0]:e}"""

        logger.info(line)

    @torch.no_grad()
    def reconstruct(self, xs, epoch, mode='valid'):
        model = self.model_ema if 'ema' in mode else self.model
        model.eval()

        xs_real = xs[:16]
        xs_recon = model(xs_real)[0]
        xs_real, xs_recon = model.module.get_recon_imgs(xs_real, xs_recon)

        grid = torch.cat([xs_real[:8], xs_recon[:8], xs_real[8:], xs_recon[8:]], dim=0)
        grid = torchvision.utils.make_grid(grid, nrow=8)
        self.writer.add_image('reconstruction', grid, mode, epoch)

    @torch.no_grad()
    def reconstruct_partial_codes(self, xs, epoch, code_idx, mode='valid', decode_type='select'):
        r"""
        Reconstruct input image using partial codebooks.
        Arguments
            xs (Tensor): input to be reconstructed
            epoch (int): the number of epoch for logging
            code_idx (int): the index of a codebook for reconstruction. (see decode_type)
            mode (string): train/valid/valid_ema for logging
            decode_type (string): ``'select'`` or ``'add'``
                If 'select', only the `code_idx`-th codebook is selected for reconstruction.
                If 'add', [0, 1, ..., code_idx] codebooks are added for reconstruction.
        """
        model = self.model_ema if 'ema' in mode else self.model
        model.eval()
        model_fn = model if not hasattr(model, 'module') else model.module

        xs_real = xs[:16]
        xs_recon = model_fn.forward_partial_code(xs_real, code_idx, decode_type)
        xs_real, xs_recon = model_fn.get_recon_imgs(xs_real, xs_recon)

        grid = torch.cat([xs_real[:8], xs_recon[:8], xs_real[8:], xs_recon[8:]], dim=0)
        grid = torchvision.utils.make_grid(grid, nrow=8)
        tag = "reconstruction_" + decode_type + f"/{code_idx}-th code"
        self.writer.add_image(tag, grid, mode, epoch)

    def save_ckpt(self, optimizer, scheduler, epoch):
        ckpt_path = os.path.join(self.config.result_path, 'epoch%d_model.pt' % epoch)
        logger.info("epoch: %d, saving %s", epoch, ckpt_path)
        ckpt = {
            'epoch': epoch,
            'state_dict': self.model.module.state_dict(),
            'discriminator': self.discriminator.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        if self.model_ema is not None:
            ckpt.update(state_dict_ema=self.model_ema.module.module.state_dict())
        torch.save(ckpt, ckpt_path)
