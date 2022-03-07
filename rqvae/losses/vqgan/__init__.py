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

from rqvae.optimizer import create_scheduler
from rqvae.optimizer.optimizer import create_resnet_optimizer

from .discriminator import NLayerDiscriminator, weights_init
from .gan_loss import hinge_d_loss, vanilla_d_loss, vanilla_g_loss
from .lpips import LPIPS


def create_vqgan_loss(loss_config):

    disc_loss_type = loss_config.disc_loss
    if disc_loss_type == "hinge":
        disc_loss = hinge_d_loss
    elif disc_loss_type == "vanilla":
        disc_loss = vanilla_d_loss
    else:
        raise ValueError(f"Unknown GAN loss '{disc_loss_type}'.")

    gen_loss_type = loss_config.gen_loss
    if gen_loss_type == 'vanilla':
        gen_loss = vanilla_g_loss
    else:
        raise ValueError(f"Unknown GAN loss '{gen_loss_type}'.")

    perceptual_loss = LPIPS()

    return disc_loss, gen_loss, perceptual_loss


def create_discriminator_with_optimizer_scheduler(disc_config, steps_per_epoch, max_epoch, distenv=None):
    model = NLayerDiscriminator(input_nc=disc_config.arch.in_channels,
                                n_layers=disc_config.arch.num_layers,
                                use_actnorm=disc_config.arch.use_actnorm,
                                ndf=disc_config.arch.ndf,
                                ).apply(weights_init)

    optimizer = create_resnet_optimizer(model, disc_config.optimizer)
    scheduler = create_scheduler(optimizer,
                                 config=disc_config.optimizer.warmup,
                                 steps_per_epoch=steps_per_epoch,
                                 max_epoch=max_epoch,
                                 distenv=distenv)

    return model, optimizer, scheduler
