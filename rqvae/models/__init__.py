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

from .ema import ExponentialMovingAverage
from .rqvae import get_rqvae
from .rqtransformer import get_rqtransformer


def create_model(config, ema=False):
    model_type = config.type.lower()
    
    if model_type == 'rq-transformer':
        model = get_rqtransformer(config)
        model_ema = get_rqtransformer(config) if ema else None
    elif model_type == 'rq-vae':
        model = get_rqvae(config)
        model_ema = get_rqvae(config) if ema else None
    else:
        raise ValueError(f'{model_type} is invalid..')

    if ema:
        model_ema = ExponentialMovingAverage(model_ema, config.ema)
        model_ema.eval()
        model_ema.update(model, step=-1)

    return model, model_ema
