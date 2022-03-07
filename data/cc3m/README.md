# Download CC-3M Dataset  

To reproduce the results of RQ-Transformer trained on [CC-3M](https://ai.google.com/research/ConceptualCaptions/),  
we provide `download_cc3m.py` to download the available images of CC-3M.  

Since CC-3M datasets only provide pairs of a text caption and an image URL, you have to download all images first.  
Please follow the instructions below to successfully download the images of CC-3M and prepare its text-image pairs.



## Step 1: Download (text, image url) tsv files

First of all, you have to download `Train_GCC-training.tsv` or `Validation_GCC-1.1.0-Validation.tsv` files.  
Please download the tsv files at [the public CC-3M homepage](https://ai.google.com/research/ConceptualCaptions/download).  
The tsv files include the pairs of (text caption, image URL).

In here, we assume that the two tsv files are downloaded at   `$CC3M_ROOT_DIR/Train_GCC-training.tsv` and `$CC3M_ROOT_DIR/Validation_GCC-1.1.0-Validation.tsv`.  



## Step 2: Download images from their URL and prepare the (text, image filename) pairs.

If the tsv files are prepared, you can download the images in the tsv files.  
After you download all available images, you have to prepare the (text, filename) pairs to use `../../rqvae/txtimg_datasets/cc3m.py`.

If you want to download train images,
```
$ mkdir $CC3M_ROOT_DIR # your root directory 
$ python download_cc3m.py --split=train --save-dir=$CC3M_ROOT_DIR
$ ls $CC3M_ROOT_DIR
Train_GCC-training.tsv       train_list.txt     train
```

If you want to download validation images, 
```bash
$ python download_cc3m.py --split=val --save-dir=$CC3M_ROOT_DIR
```

`train_list.txt` and `val_list.txt` contain the pairs of (image filename, text).  
For example, when `$CC3M_ROOT_DIR=./`, the validation (image filename, text) pairs are saved as below.
```
./val/246f243992061b252d986b1c2e0cebba  author : a life in photography -- in pictures
./val/6a8701ad0c70e74b243acade5bb90870  photograph of the sign being repaired by brave person
./val/d4473adfd46c43218ae4774dbbbe8b12  the player staring intently at a computer screen .
./val/9d487a6594f2fde0b759cd67ae9d63fa  the - bedroom stone cottage can sleep people
./val/f2c84277d1878b1285bcd4637f2df3e8  party in the park under cherry blossoms

```

After downloading the images and preparing `train_list.txt` and `val_list.txt`,   
you can use `$CC3M_ROOT_DIR` as the `root` in `../../rqvae/txtimg_datasets/cc3m.py`.  