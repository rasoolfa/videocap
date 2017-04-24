--[[
    This is reponsible to load data from h5 which was preprocessed in the python and convert each video into 
    Avg img
  ]]

require 'torch'
require 'hdf5'

local utils = require 'misc.utils_helper'
local net_utils = require 'misc.net_utils'


local VideoToAvgPool = torch.class('VideoToAvgPool') 

function VideoToAvgPool:__init(kwargs)

	--Extract parameters from kwargs
	local video_h5_path = utils.get_kwarg(kwargs, 'input_h5')
  self.tiny_save = utils.get_kwarg(kwargs, 'tiny_save', 0)
  --local new_name = utils.create_newFileName(video_h5_path, self.tiny_save .. '_IMG_' )
  local cnn_name = utils.get_kwarg(kwargs, 'cnn_name')
  local new_name = utils.create_newFileName_v2(video_h5_path, self.tiny_save .. '_IMG_', cnn_name)

  self.img_h5_path = new_name
  self.isFileThere = utils.file_exits(self.img_h5_path) --Check if the file there, don't do anything 

  -- Start init this layer parameters
  self.h5_file = hdf5.open(video_h5_path, 'r')

  -- Now time to exract info for videos/images
  -- For this version, each row in h5 database for 'videos' has the following format
  -- N,frame_rate, num_channel, img_size, img_size
  local vid_dim = self.h5_file:read('/videos'):dataspaceSize()
  assert(#vid_dim == 5, '/Video should be a 5D tensor')
  self.num_vids   = vid_dim[1]
  self.num_fm     = vid_dim[2] -- num frames 
  self.num_ch     = vid_dim[3]
  self.fm_width   = vid_dim[4]
  self.fm_height  = vid_dim[5]
  self.fm_size    = vid_dim[4]
  self.num_imgs   = self.num_vids * self.num_fm  --num_frames * num_videos
  assert(self.frame_width  == self.frame_height, 'Width and height must match')

  -- Load the captions info
  self.labels = self.h5_file:read('labels'):all()
  self.label_start_ix = self.h5_file:read('/label_start_idx'):all()
  self.label_end_ix   = self.h5_file:read('/label_end_idx'):all()
  self.label_len      = self.h5_file:read('/label_len'):all() --label_len keeps track of each caption length.

  --Print of some statistics to make sure everything look good
  print("--------------VideoToAvgPool initialization--------------")
  print("Input parameters to the VideoToAvgPool", kwargs)

  local txt0 = string.format("Input dataset has (%dx%dx%dx%dx%d) dimensions",
    vid_dim[1], vid_dim[2], vid_dim[3], vid_dim[4], vid_dim[5])
  local txt1 = string.format("This dataset contains %d videos with %d frames, %d channels, %d frame_size, and total of %d images",
    self.num_vids,self.num_fm,self.num_ch ,self.fm_size,self.num_imgs )
  print(txt0)
  print(txt1)
  print("h5 new file for video features:", self.img_h5_path)
  utils.print_separator()
end

function VideoToAvgPool:compute_cnn_feat(opt, gpu_params)
  --[[
       This function calculates CNN feastures and save them 
    ]]
    --Check if the file there, don't do anything 
    if (self.isFileThere == true) then
      print("The file "..self.img_h5_path .. " is already created.")
      print("If it needs to reprocess and recreate this file, delete manually "..self.img_h5_path .." and re-run this code.")
      return self.img_h5_path 
    end
 
    --load CNN net 
    print(string.format("Start extracting features from CNN(%s)...",opt.cnn_name))

    print(string.format("Initializing/Building CNN(%s)",opt.cnn_name))
    _,cnn_net = net_utils.build_cnn(opt, gpu_params)
    -- The VGG net has couple of dropouts, so this one MUST be set the get consistent results
    cnn_net:evaluate() 
    print(string.format("Done with CNN(%s) initialization",opt.cnn_name))

    local chunked_h5_file = hdf5.open(self.img_h5_path, 'w')

    --Write info to hd5 file
    chunked_h5_file:write("num_vids", torch.Tensor(1):fill(self.num_vids))  
    chunked_h5_file:write("num_fm", torch.Tensor(1):fill(self.num_fm))  
    chunked_h5_file:write("num_ch", torch.Tensor(1):fill(self.num_ch))  
    chunked_h5_file:write("fm_width", torch.Tensor(1):fill(self.fm_width))  
    chunked_h5_file:write("fm_height", torch.Tensor(1):fill(self.fm_height))  
    chunked_h5_file:write("num_imgs", torch.Tensor(1):fill(self.num_imgs))  

    --Write caption info to hd5 file 
    chunked_h5_file:write("label_start_idx", self.label_start_ix)  
    chunked_h5_file:write("label_end_idx", self.label_end_ix )
    chunked_h5_file:write("label_len", self.label_len )
    chunked_h5_file:write("labels", self.labels )

    print("Start processing the files, it will take a while...")

    for i = 1,  self.num_vids do
      ------------------------------
      -- Read Each video from h5 then forward through CNN
      ------------------------------
      local one_vid = self.h5_file:read('/videos'):partial({i, i}, {1, self.num_fm}, {1, self.num_ch},{1, self.fm_size }, {1, self.fm_size }) 
      one_vid = torch.squeeze(one_vid) -- input 1*16*3*256*256 --> output 16*3*256*256 

      local vid_processed = net_utils.preprocess_imgs(one_vid, gpu_params, opt) -- input 16*3*256*256  --> output 16*3*224*224  
      -- depend on the layer, the output is n*num_fm, 
      -- Example 1: i.e. 4096*16 --input 16*3*224*224 --> output 16*4096 
      -- Example 2: input 16*3*224*224 --> output 16*512*14*14
      local cnn_feat = cnn_net:forward(vid_processed) 
      local mean_feat = torch.mean( cnn_feat, 1) ---input 16*4096 --> output 1*4096

      if (self.tiny_save == 1) then
        -- tiny save just save avg features 
        chunked_h5_file:write('vid_avg'..tostring(i), mean_feat:type(utils.reset_type()))  ---save 1*4096
      elseif (self.tiny_save == 2) then
        -- tiny save just save cnn features
        chunked_h5_file:write('vid_feat'..tostring(i), cnn_feat:type(utils.reset_type())) --save  16*4096
      elseif (self.tiny_save == 3) then -- this if, only dump the convalution features
        -- tiny save just save cnn features
        chunked_h5_file:write('vid_conv'..tostring(i), cnn_feat:type(utils.reset_type())) --save  16*512*14*14
      else
        chunked_h5_file:write('vid_r'..tostring(i), one_vid:type(utils.reset_type()))   -- save  16*3*256*256
        chunked_h5_file:write('vid_p'..tostring(i), vid_processed:type(utils.reset_type()))  --save 16*3*224*224
        chunked_h5_file:write('vid_feat'..tostring(i), cnn_feat:type(utils.reset_type())) -- save 16*4096
        chunked_h5_file:write('vid_avg'..tostring(i), mean_feat:type(utils.reset_type()))  ---input 1*4096
      end  
      
      -- show some progress
      if i % 15 == 0 then
        print(string.format("%d out of %d have been processed" , i , self.num_vids))
      end

    end
    chunked_h5_file:close()
    print(string.format("Done with feature extraction, results are saved in: %s", self.img_h5_path ))

    -- returns new file name 
    return self.img_h5_path 

  end

      
      










  