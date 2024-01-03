# How to Download the Datasets

In this document, we introduce how to prepare the datasets used in this study.  
When you use the pre-computed features to evaluate FID and IS, note that you do not need to download the datasets below.  
After you download each dataset, please use the directory path, which includes the datasets below, as `root` for the `Dataset` classes in `../rqvae/img_datasets` and `../rqvae/txtimg_datasets`.  
If you have already downloaded a dataset, you can use its path as `root` for the `Dataset` classes.


## FFHQ
Before you download the FFHQ dataset, you can refer to the details in [the official repository](https://github.com/NVlabs/ffhq-dataset).  
You can download the zip file for FFHQ images 1024x1024 at [this link](https://drive.google.com/file/d/1WvlAIvuochQn_L_f9p3OdFdTiSLlnnhv/view?usp=sharing).   
After downloading the zip file, please unzip it into the `root` directory for `class FFHQ` in `../rqvae/img_datasets/ffhq.py`. 


## LSUN-{Church, Bedroom}
Before you download LSUN-{Church, Bedroom}, you can refer to the details in [the official repository](https://github.com/fyu/lsun).  
After cloning the official LSUN repository, you can easily download the two datasets using the scripts below.  

```bash
git clone https://github.com/fyu/lsun.git
cd lsun
python3 download.py -c church_outdoor -o $CHURCH_DIR_FOR_ROOT # your root directory
python3 download.py -c bedroom -o $BEDROOM_DIR_FOR_ROOT # your root directory
```

## LSUN-Cat
To download the LSUN-Cat dataset, you can refer to [the official LSUN homepage](http://dl.yf.io/lsun/objects/).  
Otherwise, use the codes below to download `cat.zip` and unzip it.  
```bash
mkdir $CAT_DIR_FOR_ROOT # your root directory
cd $CAT_DIR_FOR_ROOT
wget http://dl.yf.io/lsun/objects/cat.zip
unzip cat.zip
```
If `$CAT_DIR_FOR_ROOT` does not exist, make `$CAT_DIR_FOR_ROOT` first.

## ImageNet
For ImageNet, we use `torchvision.datasets.ImageNet` in this repository.  
Since the ImageNet dataset is no longer publicly accessible, please download the train/val [datasets](https://image-net.org/download.php).  
Then, move the train/val datasets into a directory, which is used for `root` for `torchvision.datasets.ImageNet`.  

## Conceptual Captions (CC-3M)
For the CC-3M dataset, only Image URLs are provided instead of the image file.  
To download the images and prepare (image_path, text) pairs, please refer to `./cc3m/README.md`.  


## MS-COCO
You have to make a `$COCO_ROOT_DIR` directory.  Then, make `$COCO_ROOT_DIR/ìmages` and `$COCO_ROOT_DIR/annotations` for downloading images and annotations, respectively.  
```bash
mkdir $COCO_ROOT_DIR # your root directory
cd $COCO_ROOT_DIR
mkdir images
mkdir annotations
```

You can download the images and annotations at [the official homepage](http://images.cocodataset.org/zips/train2014.zip).
Of course, you can use the scripts below.

- To download MS-COCO images
```bash
cd $COCO_ROOT_DIR/images
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014zip
```

- To download MS-COCO annotations
```bash
cd $COCO_ROOT_DIR/annotations
wget https://twg.kakaocdn.net/brainrepo/etc/RQVAE/54599b4b2286fdc2252d927aa3fd55eb/captions_val2014_30K_samples.json
```

