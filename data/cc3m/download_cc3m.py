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

from multiprocessing import Pool
import os
import io
import requests
import logging
import hashlib

import argparse
from PIL import Image
import pandas as pd
from torchvision.transforms import functional as F
from tqdm import tqdm

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--split', type=str, default='val', help='split of cc3m: train or val')
    parser.add_argument('-dir', '--save-dir', type=str, default=None, help='dir path for downloading images')
    return parser

parser = get_parser()
args = parser.parse_args()
if args.save_dir is None:
    current_dir = os.getcwd()
else:
    current_dir = args.save_dir
base_dir = os.path.join(current_dir, args.split)

assert args.split in ['train', 'val']
if args.split == 'train':
    file_path = os.path.join(current_dir, 'Train_GCC-training.tsv')
    assert os.path.exists(file_path), 'download tsv files from https://ai.google.com/research/ConceptualCaptions/download'
else:
    file_path = os.path.join(current_dir, 'Validation_GCC-1.1.0-Validation.tsv')
    assert os.path.exists(file_path), 'download tsv files from https://ai.google.com/research/ConceptualCaptions/download'

os.makedirs(base_dir, exist_ok=True)
print(f'Images are downloaded into {base_dir}')

# set up url requests
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

# Load data
print(f'Load tsv file: {file_path}')
df = pd.read_csv(file_path, delimiter='\t', header=None)

url_caption_list = [(url, caption) for index, caption, url in df.itertuples()]
print(f'Loaded {len(url_caption_list)} urls')

def download_url_with_hashing(url_caption):
    try:
        url, caption = url_caption
        filename = hashlib.md5(url.encode('utf-8')).hexdigest()
        filepath = os.path.join(base_dir, filename)  # concat to get filepath
        if not os.path.isfile(filepath):
            req = requests.get(url, stream=True, timeout=3, verify=False).raw
            image = Image.open(req).convert('RGB')

            min_image_size = 346
            new_size = image.size
            if min(new_size) > min_image_size:
                ratio = min(new_size) / min_image_size
                new_size = [int(x / ratio) for x in new_size]
                image = image.resize(new_size,)
            image.save(filepath, 'jpeg')  # save PIL image
            return 0, caption, os.path.join('./', args.split, filename)
        return 0, caption, os.path.join('./', args.split, filename)
    except Exception as e:
        url, caption = url_caption
        print(" ".join(repr(e).splitlines()))
        print(url)
        return 1, caption, url

with Pool(128) as p:
    retcodes = []
    for retcode in tqdm(p.imap_unordered(download_url_with_hashing, url_caption_list), total=len(url_caption_list)):
        retcodes.append(retcode)
    print('Download DONE')


okay_count = 0
print(f"Write (caption filename) tsv files into {args.split}_list.txt")
with open(os.path.join(current_dir, f'{args.split}_list.txt'), 'w') as f:
    with open(os.path.join(current_dir, f'{args.split}_error_list.txt', 'w')) as fnot:
        for retcode, text, imgpath in tqdm(retcodes, total=len(retcodes)):
            if retcode == 0:
                okay_count += 1
                f.write(f'{imgpath}\t{text}\n')
            else:
                fnot.write(f'{imgpath}\t{text}\n')

print(f"Total {okay_count} / {len(retcodes)} pairs are prepared.")
