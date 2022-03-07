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

import io
import os
from pathlib import Path
import pickle
import string
from typing import Tuple, Any

import torchvision
import lmdb
from PIL import Image


class LSUNClass(torchvision.datasets.VisionDataset):

    subpaths = {'church': 'church/church_outdoor_train_lmdb',
                'church_val': 'church/church_outdoor_val_lmdb',
                'bedroom': 'bedroom/bedroom_train_lmdb',
                'bedroom_val': 'bedroom/bedroom_val_lmdb',
                'cat': 'cat',
                }
    valid_categories = ['church', 'bedroom', 'cat']

    def __init__(self, root, category_name='church', transform=None):

        assert category_name in LSUNClass.valid_categories
        root = Path(root) / LSUNClass.subpaths[category_name]
        print(root)

        super(LSUNClass, self).__init__(root, transform=transform)

        self.env = lmdb.open(root, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()["entries"]
        cache_file = "_cache_" + "".join(c for c in root if c in string.ascii_letters)
        cache_file = os.path.join(root, cache_file)
        if os.path.isfile(cache_file):
            self.keys = pickle.load(open(cache_file, "rb"))
        else:
            with self.env.begin(write=False) as txn:
                self.keys = [key for key in txn.cursor().iternext(keys=True, values=False)]
            pickle.dump(self.keys, open(cache_file, "wb"))

        self.exception_idx = [29343, 88863] if category_name == 'cat' else []

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        index = index if index not in self.exception_idx else index - 1

        img, target = None, None
        env = self.env
        with env.begin(write=False) as txn:
            imgbuf = txn.get(self.keys[index])

        buf = io.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, 0

    def __len__(self) -> int:
        return self.length
