# videocap
Memory-augmented Attention Modelling for Videos

Step 0) Install require packages

sudo add-apt-repository ppa:mc3man/trusty-media
sudo apt-get update 
sudo apt-get dist-upgrade
sudo apt-get install ffmpeg python-opencv
sudo pip install scipy numpy

--install opencv
It is better to install opencv from source not from repro
sudo apt-get install python-opencv  


--install torch 
luarocks install torch && luarocks install image && luarocks install sys && luarocks install nn && luarocks install optim && luarocks install lua-cjson && luarocks install cutorch  && luarocks install cunn  && luarocks install loadcaffe

Step 1:
Download Data from 
http://upplysingaoflun.ecn.purdue.edu/~yu239/datasets/youtubeclips.zip

Download  VGG16 pretrained model in ~/Data/vgg_pre:
http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel

Step 2) unzip data:
unzip youtubeclips.zip
let's assume data are in ~/Data/youtubeclips-dataset

Step 3) Prepare data [it takes a couple of hours]
-create the following folders
mdkir ~/Data/YouTubeClip_mp4
mkdir ~/Data/Youtube_frames_8 
mkdir ~/Data/Y_8_data

python -u scripts/convert_aviTompg.py --video_dir ~/Data/youtubeclips-dataset --output ~/Data/YouTubeClip_mp4 
python -u scripts/build_frames.py  --clip_dir ~/Data/YouTubeClip_mp4  --output ~/Data/Youtube_frames_8 --num_frames 8 --frame_type continuous 

Step  4) Preporcess Data

python -u scripts/data_prep.py --frame_dir ~/Data/Youtube_frames_8 --input_json Youtube/YT_40_raw_all.json --max_length 30 --output_json ~/Data/Y_8_data/YT_8_len30.json --output_h5 ~/Data/Y_8_data/YT_8_len30.h5 --dataset_name YT_all --only_test 0 --word_count_threshold 0 


step 5) Train the model and Report results

CUDA_VISIBLE_DEVICES=1 th train_SeqToSeq_MeMLocSoft_R2.lua -grad_clip 2 -checkpoint_every 600 -embed_size 402 -gpu_id 0 -mem_size 797 -num_layers 2 -sample_max 1 -init_type_loc nnet -lang_eval_method METEOR -cnn_proto ~/Data/vgg_pre/VGG_ILSVRC_19_layers_deploy.prototxt -cnn_layer_name relu5_4 -cnn_name VGG-16 -weightDecayOptim 1e-05 -learning_rate 2e-05 -lr_decay_factor 0.1 -input_h5 ~/Data/Y_8_data/YT_8_len30.h5 -optim adam -json_file ~/Data/Y_8_data/YT_8_len30.json -layer_num 42 -adam_beta1 0.8 -batch_size 16 -f_gt Youtube/YT_40_captions_val.json -adam_beta2 0.999 -dropout 0.5 -lr_decay_every 200 -gpu_backend cuda -lr_cap 1e-10 -rnn_size 1479 -checkpoint_name ~/Data/cv/yt_n -dname yt -log_id mylog_mlsnnet_y_w11111 -cnn_model ~/Data/vgg_pre/VGG_ILSVRC_19_layers.caffemodel -max_epoch 400 -optim_epsilon 1e-08


CUDA_VISIBLE_DEVICES=1 th eval_SeqToSeq_MemLocSoft_R2.lua -gpu_id 0 -split test -input_h5 ~/Data/Y_8_data/YT_8_len30.h5 -json_file ~/Data/Y_8_data/YT_8_len30.json -f_gt Youtube/YT_40_captions_test.json -gpu_backend cuda  -checkpoint_name ~/Data/cv/yt_test  -init_from /Data/cv/yt_n/mylog_mlsnnet_y_w11111.t7






 
