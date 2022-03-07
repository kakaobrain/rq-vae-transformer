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

import numpy as np
import torch
from torch.nn import functional as F

import clip
from PIL import Image

from .fid import create_dataset_from_files

from rqvae.txtimg_datasets.cc3m import Cc3mRawTextOnly
from rqvae.txtimg_datasets.coco import CocoRawTextOnly


def get_clip():
    model_clip, preprocess_clip = clip.load("ViT-B/32", device='cpu')
    return model_clip, preprocess_clip


@torch.no_grad()
def clip_score(pixels, texts, model_clip, preprocess_clip, device=torch.device('cuda')):
    # pixels: 0~1 valued tensors
    pixels = pixels.cpu().numpy()
    pixels = np.transpose(pixels, (0, 2, 3, 1))

    images = [preprocess_clip(Image.fromarray((pixel*255).astype(np.uint8))) for pixel in pixels]
    images = torch.stack(images, dim=0).to(device=device)
    texts = clip.tokenize(texts).to(device=device)

    image_features = model_clip.encode_image(images)
    text_features = model_clip.encode_text(texts)

    scores = F.cosine_similarity(image_features, text_features).squeeze()

    return scores


def compute_clip_score(fake_path,
                       dataset_name='cc3m',
                       dataset_root=None,
                       split='val',
                       batch_size=100,
                       device=torch.device('cuda'),
                       ):

    model_clip, preprocess_clip = get_clip()
    model_clip.to(device=device)
    model_clip.eval()

    img_dataset = create_dataset_from_files(fake_path)

    if dataset_name == 'cc3m':
        root = dataset_root if dataset_root else 'data/cc3m'
        txt_dataset = Cc3mRawTextOnly(root, split=split)
    elif dataset_name == 'coco':
        root = dataset_root if dataset_root else 'data/coco'
        txt_dataset = CocoRawTextOnly(root, split=split)
    else:
        raise ValueError(f'Unsupported dataset: {dataset_name}')

    # Here we assume that the order of imgs is same as the order of txts,
    # possibly has some duplicates at the end due to the distributed sampler.
    assert len(img_dataset) >= len(txt_dataset)
    img_dataset = torch.utils.data.Subset(img_dataset, np.arange(len(txt_dataset)))

    img_loader = torch.utils.data.DataLoader(img_dataset, batch_size=batch_size)
    txt_loader = torch.utils.data.DataLoader(txt_dataset, batch_size=batch_size)

    scores = []
    for (imgs,), txts in zip(img_loader, txt_loader):
        score = clip_score(imgs, txts, model_clip, preprocess_clip)
        scores.append(score.cpu().numpy())

    scores = np.concatenate(scores)
    scores_avg = scores.mean()

    return scores_avg
