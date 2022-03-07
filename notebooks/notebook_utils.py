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
import sys

from PIL import Image
import yaml
import numpy as np
import torch
import torchvision
import clip
import torch.nn.functional as F

from rqvae.utils.config import load_config, augment_arch_defaults
from rqvae.models import create_model
from rqvae.txtimg_datasets.tokenizers import create_tokenizer


class TextEncoder:
    def __init__(self, tokenizer_name, context_length=64, lowercase=True):
        self.tokenizer = create_tokenizer(tokenizer_name, lowercase=lowercase)
        self.context_length = context_length
        
        
        self.tokenizer.add_special_tokens(["[PAD]"])
        self.tokenizer.enable_padding(length=self.context_length,
                                      pad_id=self.tokenizer.token_to_id("[PAD]"))
        self.tokenizer.enable_truncation(max_length=self.context_length)
    
    def encode(self, texts):
        output = self.tokenizer.encode(texts)
        ids = output.ids
        
        if not isinstance(ids, torch.LongTensor):
            ids = torch.LongTensor(ids)
            
        return ids
    
    def __call__(self, texts):
        return self.encode(texts)
    

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

def get_initial_sample(batch_sample_shape, device=torch.device('cuda')):
    partial_sample = torch.zeros(*batch_sample_shape, 
                                 dtype=torch.long, 
                                 device=device)
    return partial_sample
    
@torch.no_grad()
def get_clip_score(pixels, texts, model_clip, preprocess_clip, device=torch.device('cuda')):
    # pixels: 0~1 valued tensors
    pixels = pixels.cpu().numpy()
    pixels = np.transpose(pixels, (0, 2, 3, 1))

    images = [preprocess_clip(Image.fromarray((pixel*255).astype(np.uint8))) 
              for pixel in pixels]
    images = torch.stack(images, dim=0).to(device=device)
    texts = clip.tokenize(texts).to(device=device)

    image_features = model_clip.encode_image(images)
    text_features = model_clip.encode_text(texts)

    scores = F.cosine_similarity(image_features, text_features).squeeze()

    return scores

@torch.no_grad()
def get_generated_images_by_texts(model_ar,
                                  model_vqvae,
                                  text_encoder,
                                  model_clip,
                                  preprocess_clip,
                                  text_prompts,
                                  num_samples,
                                  temperature,
                                  top_k,
                                  top_p,
                                  amp=True,
                                  fast=True,
                                  is_tqdm=True,
                                 ):
    
    sample_shape = model_ar.get_block_size()
    
    text_cond = text_encoder(text_prompts).unsqueeze(0).repeat(num_samples, 1).cuda()
    
    initial_codes = get_initial_sample([num_samples, *sample_shape])
    generated_codes = model_ar.sample(initial_codes,
                                      model_vqvae,
                                      cond=text_cond,
                                      temperature=temperature,
                                      top_k=top_k,
                                      top_p=top_p,
                                      amp=amp,
                                      fast=fast,
                                      is_tqdm=is_tqdm,
                                     )
    pixels = torch.cat([model_vqvae.decode_code(generated_codes[i:i+1]) 
                                        for i in range(generated_codes.size(0))
                   ], dim=0)

    clip_scores = get_clip_score(pixels, 
                                 text_prompts, 
                                 model_clip, 
                                 preprocess_clip,
                                )

    reranked_idxs = clip_scores.argsort(descending=True)
    reranked_pixels = pixels[reranked_idxs]
    
    return reranked_pixels
