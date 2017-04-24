########################################
#  This code extract frames from a video clip.
#  In this code. opencv has been used, so opencv needed to be installed
#  sudo apt-get install python-opencv
########################################

from __future__ import print_function
import argparse
import cv2
import os
import numpy as np
from  utils import get_list_files

def extract_frames(clip_dir, output_dir, num_frames = 16, frame_type = 'continuous', stride = 30, f_ext ='mp4', stride_type = 'fps' ):
    """
     This function processes the input clips and extract some frames.
     input parameters:
         num_frames: number of frames to be extracted
         frame_type: continuous or random. if random, it doesn't use the stride
         clip_dir: input dir
         output_dir: output directory to save the results
         stride: is used to move between frames, only consider when frame_type is continuous
         stride: can be override by the frame_rates.
    """
    clip_list = get_list_files(clip_dir, f_ext = f_ext )
    prng = np.random.RandomState(1234) # make reproducible
    print("Number of video: %d" %(len(clip_list)))
    stride_original =  stride
    for clip in clip_list:
        
        frames_f_name = clip.split('_')[0]
        # if file is already created just ignore the reset
        if os.path.exists(output_dir +"/" + frames_f_name + "_f.npz") and os.path.isfile(output_dir +"/" + frames_f_name + "_f.npz"):
            print("clip %s is already processed" %(clip))
            continue

        print(clip)
        #extract clip info
        cur_vidcap = cv2.VideoCapture(clip_dir + "/" + clip)
        clip_len = int(cur_vidcap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        width = int(cur_vidcap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        height = int(cur_vidcap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(cur_vidcap.get(cv2.cv.CV_CAP_PROP_FPS))

        images = np.zeros((num_frames,height, width,3),dtype='uint8')
        if clip_len <= 0:
             print("***Error*** Proceesing %s which has %d frames and size(%d, %d)" % (clip, clip_len, height, width ))
             continue

        if frame_type == 'rand_sample':  #uniform sampling, but keep the order of frames

            print("Proceesing %s which has %d frames and size(%d, %d)." % (clip, clip_len, height, width))
            if  num_frames > clip_len - 5:
                print("##Using sampleing with replacement, Proceesing %s which has %d frames and size(%d, %d)" % (clip, clip_len, height, width ))
                idx_f = prng.choice(clip_len - 5 , num_frames, replace=True)
            else:
                idx_f = prng.choice(clip_len - 5 , num_frames, replace=False)       
            idx_f.sort() #  when need the lsit to be in order 
            print("rand_sample:", idx_f)

        elif frame_type == 'rand_order':   #uniform sampling, but randomize the order

            print("Proceesing %s which has %d frames and size(%d, %d)." % (clip, clip_len, height, width)) 
            if  num_frames > clip_len - 5:
                print("##Using sampleing with replacement, Proceesing %s which has %d frames and size(%d, %d)" % (clip, clip_len, height, width ))
                idx_f = prng.choice(clip_len - 5 , num_frames, replace=True)
            else:
                idx_f = prng.choice(clip_len - 5 , num_frames, replace=False)    
            print("rand_order:", idx_f)

        else:

            if (frame_rate > 0  and frame_rate < clip_len and stride_type == 'fps'):
                stride =  frame_rate * 2
                print("##Stride %d, Proceesing %s which has %d frames and size(%d, %d) with fps %d" % (stride, clip, clip_len, height, width, frame_rate))

            if(clip_len < num_frames * stride ): # in case where there is less frame than num_frames * stride
                stride =  clip_len / num_frames
                print("##UPDATED Stride %d, Proceesing %s which has %d frames and size(%d, %d)" % (stride, clip, clip_len, height, width ))
            else:
                print("Proceesing %s which has %d frames and size(%d, %d). Stride size is %d." % (clip, clip_len, height, width, stride ))    
            
            if(num_frames > clip_len):
                print("******* not enough frame for %s" %(clip)) 
                stride = 1  
                print("##UPDATED Stride %d, Proceesing %s which has %d frames and size(%d, %d)" % (stride, clip, clip_len, height, width ))

        start_f = 0
        for i in range(num_frames):
            if frame_type == 'rand_sample' or frame_type == 'rand_order':
               cur_vidcap.set(1,idx_f[i])
            else: 
                if ( start_f + stride == clip_len):
                    print("***start_f is adjusted***")
                    start_f = start_f -1
                cur_vidcap.set(1,start_f)
                start_f =  start_f + stride 

            success, img = cur_vidcap.read()
            images [i] = img
        stride = stride_original

        #print(clip, frames_f_name, width, height)
        np.savez_compressed(output_dir +"/" + frames_f_name + "_f.npz",images)
        cur_vidcap.release()  
        print("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--clip_dir', required=True, help='source video files dir')
    parser.add_argument('--output', required=True, help='dest folder for frames files')
    parser.add_argument('--clip_format', default = 'mp4',help='clips encoding format')

    # to extract frame, the program can extract frames from continously or randomly from all over the clip 
    parser.add_argument('--frame_type', default='continuous', help='rand_order|rand_sample|continuous')
    parser.add_argument('--num_frames', type=int, default=16, help='number of frame to extract per clip 1-30')
    parser.add_argument('--stride', type=int, default=30, help='stride btween two frames')
    parser.add_argument('--stride_type', default='fps', help='stride btween two frames will frame_rates*2')


    args = parser.parse_args()
    print("\nParams:", vars(args),"\n")
    extract_frames(args.clip_dir, args.output, num_frames = args.num_frames, frame_type = args.frame_type, stride = args.stride, f_ext = args.clip_format, stride_type = args.stride_type)
