# Introduction
This repository implements the method which is presented in the following paper:
- [Memory-augmented Attention Modelling for Videos] (https://arxiv.org/abs/1611.02261/)

If you find this code useful in your research, please cite:
```
@article{Fakoor16,
  author    = {Rasool Fakoor and
               Abdel{-}rahman Mohamed and
               Margaret Mitchell and
               Sing Bing Kang and
               Pushmeet Kohli},
  title     = {Memory-augmented Attention Modelling for Videos},
  journal   = {CoRR},
  volume    = {abs/1611.02261},
  year      = {2016},
  url       = {http://arxiv.org/abs/1611.02261},
}
```

## Code setup

### Step 0) Install required packages

sudo add-apt-repository ppa:mc3man/trusty-media
sudo apt-get update 
sudo apt-get dist-upgrade
sudo apt-get install ffmpeg python-opencv
sudo pip install scipy numpy

#### install opencv
It is better to install opencv from source not from repro
sudo apt-get install python-opencv  


#### install torch 
luarocks install torch && luarocks install image && luarocks install sys && luarocks install nn && luarocks install optim && luarocks install lua-cjson && luarocks install cutorch  && luarocks install cunn  && luarocks install loadcaffe

#### Add coco-caption eval codes 
Go to https://github.com/tylin/coco-caption/tree/master/pycocoevalcap
Download the following folders and add them to eval_caption/
- bleu/
- cider/
- meteor/
- rouge/
- tokenizer/

#### Step 1)
Download Data from 
http://upplysingaoflun.ecn.purdue.edu/~yu239/datasets/youtubeclips.zip

Download  VGG16 pretrained model in ~/Data/vgg_pre:
http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel

#### Step 2) unzip data:
unzip youtubeclips.zip
let's assume data are in ~/Data/youtubeclips-dataset

#### Step 3) Prepare data [it takes a couple of hours]
-create the following folders
mdkir ~/Data/YouTubeClip_mp4
mkdir ~/Data/Youtube_frames_8 
mkdir ~/Data/Y_8_data

python -u scripts/convert_aviTompg.py --video_dir ~/Data/youtubeclips-dataset --output ~/Data/YouTubeClip_mp4 
python -u scripts/build_frames.py  --clip_dir ~/Data/YouTubeClip_mp4  --output ~/Data/Youtube_frames_8 --num_frames 8 --frame_type continuous 

#### Step 4) Preprocess Data

python -u scripts/data_prep.py --frame_dir ~/Data/Youtube_frames_8 --input_json Youtube/YT_40_raw_all.json --max_length 30 --output_json ~/Data/Y_8_data/YT_8_len30.json --output_h5 ~/Data/Y_8_data/YT_8_len30.h5 --dataset_name YT_all --only_test 0 --word_count_threshold 0 


#### Step 5) Train the model and Report results

CUDA_VISIBLE_DEVICES=0 th train_SeqToSeq_MeMLocSoft_R2.lua -cnn_proto ~/Data/vgg_pre/VGG_ILSVRC_19_layers_deploy.prototxt -input_h5 ~/Data/Y_8_data/YT_8_len30.h5 -json_file ~/Data/Y_8_data/YT_8_len30.json  -f_gt Youtube/YT_40_captions_val.json  -checkpoint_name ~/Data/cv/yt_n -log_id mylog_mlsnnet_y_w11111 -cnn_model ~/Data/vgg_pre/VGG_ILSVRC_19_layers.caffemodel 


CUDA_VISIBLE_DEVICES=0 th eval_SeqToSeq_MemLocSoft_R2.lua -gpu_id 0 -split test -input_h5 ~/Data/Y_8_data/YT_8_len30.h5 -json_file ~/Data/Y_8_data/YT_8_len30.json -f_gt Youtube/YT_40_captions_test.json -gpu_backend cuda  -checkpoint_name ~/Data/cv/yt_test  -init_from /Data/cv/yt_n/mylog_mlsnnet_y_w11111.t7


## Acknowledgements

The structure of this codebase is inspired by https://github.com/karpathy/neuraltalk2. In addation, some functions from https://github.com/karpathy/neuraltalk2 have been re-written/changed in this codebase which are [mostly] excpliclty mentioned in my code.


Please contact me (@rasoolfa) if you find a bug or problem with this code.





 
