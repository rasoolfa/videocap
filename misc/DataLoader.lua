--[[
   DataLoader is reponsible to load data from h5 which was preprocessed in the 
   python and further edited in the torch. 
  ]]

require 'torch'
require 'hdf5'

local utils = require 'misc.utils_helper'

local DataLoader = torch.class('DataLoader') 

function DataLoader:__init(kwargs)

	--Extract parameters from kwargs
	local img_h5_path = utils.get_kwarg(kwargs, 'img_h5_path')
  	local j_file = utils.get_kwarg(kwargs, 'json_file')
  	self.batch_size = utils.get_kwarg(kwargs, 'batch_size')
  	-- type of input can be vid_avg or vid_feat
  	self.i_type =  utils.get_kwarg(kwargs, 'input_type','vid_avg')

    -- Start init this layer parameters
    self.h5_file = hdf5.open(img_h5_path, 'r')
    self.info = utils.read_json(j_file)
    self.ix_to_word = self.info.ix_to_word
    self.vocab_size = utils.get_dict_len(self.info.ix_to_word) 

    -- Now time to exract info for videos/images
    -- For this version, each row in h5 database for 'videos' has the following format
    -- N,frame_rate, num_channel, img_size, img_size
    --[[
    	img_h5_path h5 file has the following keys realted to vidoes:
    	vid_ri : contains original raw videos: 16*3*256*256 
    	vid_pi : contains preprocessed videos: 16*3*224*224  
    	vid_feati: contains cnn features for each frame: 16*4096  
    	vid_avgi: contains averaged mean cnn features for the input videos: 1*4096    
    	where i should be replace with a number 1-6383
		- Each files for different input types, have teh followings shape:
			vid_avg:  1*4096
			vid_conv: 16*512*14*14
			vid_feat: 16*4096 
      ]]
    self.feat_size  = self.h5_file:read(self.i_type..tostring(1)):dataspaceSize()[2]
 	self.num_vids   =  self.h5_file:read('num_vids'):all()
    self.num_fm     =  self.h5_file:read('num_fm'):all()
  	self.num_ch      =  self.h5_file:read('num_ch'):all()
  	self.fm_width  =  self.h5_file:read('fm_width'):all()
  	self.fm_height =  self.h5_file:read('fm_height'):all()
  	self.num_imgs   =  self.h5_file:read('num_imgs'):all()


    --- if type is vid_conv, extract the kernel size of convolution
    if self.i_type == 'vid_conv' then
    	local temp_vid = self.h5_file:read(self.i_type..tostring(1)):all():view(self.num_fm[1],self.feat_size, -1 )
    	self.conv_loc_size = temp_vid:size(3)
    end

    assert(self.frame_width  == self.frame_height, 'Width and height must match')

    -- load in the seq data, i.e. captions
  	local seq_size = self.h5_file:read('labels'):dataspaceSize()
  	self.seq_length = seq_size[2]
  	-- Load the captions info
  	self.label_start_ix = self.h5_file:read('label_start_idx'):all()
  	self.label_end_ix   = self.h5_file:read('label_end_idx'):all()
  	self.label_len      = self.h5_file:read('label_len'):all() --label_len keeps track of each caption length.

  	--Group the data based on the split
  	self.split_ix   = {}
  	self.split_iter = {} -- iterator as the name implies, will be used to iterate ove over the current split
  	for idx, v in pairs(self.info.video) do
  		local s = v.split
  		if not self.split_ix[s] then
  			self.split_ix[s] = {}
  			self.split_iter[s] = 1
  		end
  		table.insert(self.split_ix[s], idx)
  	end

    --Print of some statistics to make sure everything look good
    print("--------------DataLoader--------------")
   	print("Input parameters to the Dataloader:", kwargs)
 	print('Sequence length in data is ' .. self.seq_length)
    local txt1 = string.format("This dataset contains %d videos with %d frames, %d channels, with frames size of %d*%d", 
    	self.num_vids[1],self.num_fm[1],self.num_ch[1] ,self.fm_width[1],self.fm_height[1])
  
    print(txt1)
    -- some statistics about the data split
    for k, v in pairs(self.split_ix) do
    	print(string.format("Number of data in %s is %d ", k, #v))
    end
    utils.print_separator()
	self.vid_batch_raw = torch.FloatTensor()
	self.label_batch = torch.LongTensor()
end

function DataLoader:get_batch_AVGfeat(opt)
	--[[
	 In this function the mini-batch will be created based on the input parameters 
	 For now, for each kind of batches, we name the function same name as the feature.
	 For example, here we use mean feature which is calculated by avg over 16 frames after cnn 
	 features have been extracted.  
	]]
	assert(self.i_type == 'vid_avg', string.format('Wrong input type ("%s"), should be:"vid_avg"',self.i_type))

	local split = utils.get_kwarg(opt, 'split')
	local batch_size  = utils.get_kwarg(opt, 'batch_size', 5)
	local seq_per_img = utils.get_kwarg(opt, 'seq_per_img', 1) -- Keep this one for future compatibility 

	-- check the current split is valid
	local l_split_ix = self.split_ix[split]
	assert(l_split_ix, string.format('The split "%s" is not found', split))
	
	local max_idx = #self.split_ix[split] -- len current split
    -- during the prediction, for each sample, we must only predict one
    -- so it is important to during test/eval, when we wrap around, adjust the batch size 
	if (split ~= 'train') then
		local diff = max_idx - self.split_iter[split] + 1
		if  diff < batch_size then
			batch_size = diff 
		end	
	end

	self.vid_batch_raw:resize(batch_size, self.feat_size):zero()
	self.label_batch:resize(batch_size * seq_per_img, self.seq_length):zero() ----
	local wrapped = false
	local meta_info = {}

	for b = 1, batch_size do

		------------------------------
		-- Handle the split index
		------------------------------
		local c_idx =  self.split_iter[split] -- current index
		local next_idx = c_idx + 1

		if next_idx > max_idx then -- if the next_index is greater the length of current split
			next_idx =1
			wrapped = true
		end
		-- update iterator
		self.split_iter[split] = next_idx
		-- get current data
		local b_ix = l_split_ix[c_idx] 
		--some sanity check
		assert(b_ix ~= nil, 'Somthing wrong with the split')

		------------------------------
		-- Now fetch the videos from the input 
		-- Each row in the h5 is just one sample
		------------------------------
		local h5_row_key = 'vid_avg'..tostring(b_ix)
		self.vid_batch_raw[b] = self.h5_file:read(h5_row_key):all():squeeze() -- The output is 1*4096, we need to make them 4096 

		------------------------------
		-- Now fetch the captions: Not happy with the code, should be written simpler
		------------------------------
		local s_ix_start = self.label_start_ix[b_ix] -- index start for current input
		local s_ix_end   =  self.label_end_ix[b_ix]  -- index end for current input 
		local num_cap_ix = s_ix_end - s_ix_start + 1 -- number of captions available for this video

		if  num_cap_ix <= 0 then 
			error(string.format('No caption for current image "%d in batch %b"', b_ix, b))
		end
        
		local seq 
		if (num_cap_ix < seq_per_img) then
			-- no enough sample
			seq = torch.LongTensor(seq_per_img, self.seq_length)
			for q = 1, seq_per_img do
				local idx_label = torch.random(s_ix_start, s_ix_end )
				seq[{ {q,q} }] = self.h5_file:read('labels'):partial({idx_label, idx_label}, {1,self.seq_length})
			end
		else
			-- enough sample
			local idx_label = torch.random(s_ix_start, s_ix_end - seq_per_img + 1  )
			seq = self.h5_file:read('labels'):partial({idx_label, idx_label + seq_per_img - 1}, {1,self.seq_length})
		end

		local bl = (b - 1) * seq_per_img + 1
		self.label_batch[{{bl, bl + seq_per_img -1}}] = seq

		------------------------------
		-- Keep some local info
		------------------------------
		local info_batch = {}
		info_batch.vid = self.info.video[b_ix].video_id	
		table.insert(meta_info, info_batch )
	end
	
	------------------------------
	-- Now make all batch one table and return it
	------------------------------
	local data = {}
	data.videos = self.vid_batch_raw
	data.labels = self.label_batch:transpose(1,2):contiguous() -- note: make label sequences go down as columns
	data.bounds = {it_pos_now = self.split_iter[split], it_max = #l_split_ix, wrapped = wrapped}
	data.info   = meta_info
	data.size   = batch_size --add this one, because for evaluation the batch_size can be adjusted depens on the input  

	return data
end

function DataLoader:get_batch_frames(opt)
	--[[
	 In this function the mini-batch will be created based on the input parameters 
	 For now, for each kind of batches, we name the function same name as the feature.
	 For example, here we use mean feature which is calculated by avg over 16 frames after cnn 
	 features have been extracted.  
	]]
	assert(self.i_type == 'vid_feat', string.format('Wrong input type ("%s"), should be:"vid_feat"',self.i_type ))  

	local split = utils.get_kwarg(opt, 'split')
	local batch_size  = utils.get_kwarg(opt, 'batch_size', 5)
	local seq_per_img = utils.get_kwarg(opt, 'seq_per_img', 1) -- Keep this one for future compatibility 

	-- check the current split is valid
	local l_split_ix = self.split_ix[split]
	assert(l_split_ix, string.format('The split "%s" is not found', split))

	local max_idx = #self.split_ix[split] -- len current split
    -- during the prediction, for each sample, we must only predict one
    -- so it is important to during test/eval, when we wrap around, adjust the batch size 
	if (split ~= 'train') then
		local diff = max_idx - self.split_iter[split] + 1
		if  diff < batch_size then
			batch_size = diff 
		end	
	end

	self.vid_batch_raw:resize(batch_size, self.num_fm[1], self.feat_size):zero()
	self.label_batch:resize(batch_size * seq_per_img, self.seq_length):zero() ----
	local wrapped = false
	local meta_info = {}

	for b = 1, batch_size do

		------------------------------
		-- Handle the split index
		------------------------------
		local c_idx =  self.split_iter[split] -- current index
		local next_idx = c_idx + 1

		if next_idx > max_idx then -- if the next_index is greater the length of current split
			next_idx =1
			wrapped = true
		end
		-- update iterator
		self.split_iter[split] = next_idx
		-- get current data
		local b_ix = l_split_ix[c_idx] 
		--some sanity check
		assert(b_ix ~= nil, 'Somthing wrong with the split')

		------------------------------
		-- Now fetch the videos from the input 
		-- Each row in the h5 is just one sample
		------------------------------
		local h5_row_key = self.i_type..tostring(b_ix)
		self.vid_batch_raw[b] = self.h5_file:read(h5_row_key):all():squeeze() -- The output is 16*4096

		------------------------------
		-- Now fetch the captions: Not happy with the code, should be written simpler
		------------------------------
		local s_ix_start = self.label_start_ix[b_ix] -- index start for current input
		local s_ix_end   =  self.label_end_ix[b_ix]  -- index end for current input 
		local num_cap_ix = s_ix_end - s_ix_start + 1 -- number of captions available for this video

		if  num_cap_ix <= 0 then 
			error(string.format('No caption for current image "%d in batch %b"', b_ix, b))
		end
        
		local seq 
		if (num_cap_ix < seq_per_img) then
			-- no enough sample
			seq = torch.LongTensor(seq_per_img, self.seq_length)
			for q = 1, seq_per_img do
				local idx_label = torch.random(s_ix_start, s_ix_end )
				seq[{ {q,q} }] = self.h5_file:read('labels'):partial({idx_label, idx_label}, {1,self.seq_length})
			end
		else
			-- enough sample
			local idx_label = torch.random(s_ix_start, s_ix_end - seq_per_img + 1  )
			seq = self.h5_file:read('labels'):partial({idx_label, idx_label + seq_per_img - 1}, {1,self.seq_length})
		end

		local bl = (b - 1) * seq_per_img + 1
		self.label_batch[{{bl, bl + seq_per_img -1}}] = seq

		------------------------------
		-- Keep some local info
		------------------------------
		local info_batch = {}
		info_batch.vid = self.info.video[b_ix].video_id	
		--info_batch.wrapped = wrapped -- this will be used in the evalution
		table.insert(meta_info, info_batch )
	end
	
	------------------------------
	-- Now make all batch one table and return it
	------------------------------
	local data = {}
	data.videos = self.vid_batch_raw
	data.labels = self.label_batch:transpose(1,2):contiguous() -- note: make label sequences go down as columns
	data.bounds = {it_pos_now = self.split_iter[split], it_max = #l_split_ix, wrapped = wrapped}
	data.info   = meta_info
	data.size   = batch_size --add this one, because for evaluation the batch_size can be adjusted depens on the input  

	return data
end

function DataLoader:get_batch_frames_Conv_AVG(opt)
	--[[
           This function reads conv features and create a mean features
	  ]]
	assert(self.i_type == 'vid_conv', string.format('Wrong input type ("%s"), should be:"vid_conv"',self.i_type ))  
 	local split = utils.get_kwarg(opt, 'split')
	local batch_size  = utils.get_kwarg(opt, 'batch_size', 5)
	local seq_per_img = utils.get_kwarg(opt, 'seq_per_img', 1) -- Keep this one for future compatibility 
	--local dtype = utils.get_kwarg(opt, 'dtype') -- Since doing mean here, lets do the mean in GPU


	-- check the current split is valid
	local l_split_ix = self.split_ix[split]
	assert(l_split_ix, string.format('The split "%s" is not found', split))

	local max_idx = #self.split_ix[split] -- len current split

    -- during the prediction, for each sample, we must only predict one
    -- so it is important to during test/eval, when we wrap around, adjust the batch size 
	if (split ~= 'train') then
		local diff = max_idx - self.split_iter[split] + 1
		if  diff < batch_size then
			batch_size = diff 
		end	
	end

	self.vid_batch_raw:resize(batch_size, self.num_fm[1], self.feat_size):zero()
	self.label_batch:resize(batch_size * seq_per_img, self.seq_length):zero() ----
	local wrapped = false
	local meta_info = {}

	for b = 1, batch_size do

		------------------------------
		-- Handle the split index
		------------------------------
		local c_idx =  self.split_iter[split] -- current index
		local next_idx = c_idx + 1

		if next_idx > max_idx then -- if the next_index is greater the length of current split
			next_idx =1
			wrapped = true
		end
		-- update iterator
		self.split_iter[split] = next_idx
		-- get current data
		local b_ix = l_split_ix[c_idx] 
		--some sanity check
		assert(b_ix ~= nil, 'Somthing wrong with the split')

		------------------------------
		-- Now fetch the videos from the input 
		-- Each row in the h5 is just one sample
		------------------------------
		local h5_row_key = self.i_type..tostring(b_ix)
		 -- The output is 16*512*14*14, we need to make them 16*512
		 -- 16*512*14*14 --> 16*512*196 --> 16*512*1 --> 16*512
		local temp_vid = self.h5_file:read(h5_row_key):all():view(self.num_fm[1],self.feat_size, -1 )
		self.vid_batch_raw[b] = torch.mean( temp_vid, 3):squeeze()

		------------------------------
		-- Now fetch the captions: Not happy with the code, should be written simpler
		------------------------------
		local s_ix_start = self.label_start_ix[b_ix] -- index start for current input
		local s_ix_end   =  self.label_end_ix[b_ix]  -- index end for current input 
		local num_cap_ix = s_ix_end - s_ix_start + 1 -- number of captions available for this video

		if  num_cap_ix <= 0 then 
			error(string.format('No caption for current image "%d in batch %b"', b_ix, b))
		end
        
		local seq 
		if (num_cap_ix < seq_per_img) then
			-- no enough sample
			seq = torch.LongTensor(seq_per_img, self.seq_length)
			for q = 1, seq_per_img do
				local idx_label = torch.random(s_ix_start, s_ix_end )
				seq[{ {q,q} }] = self.h5_file:read('labels'):partial({idx_label, idx_label}, {1,self.seq_length})
			end
		else
			-- enough sample
			local idx_label = torch.random(s_ix_start, s_ix_end - seq_per_img + 1  )
			seq = self.h5_file:read('labels'):partial({idx_label, idx_label + seq_per_img - 1}, {1,self.seq_length})
		end

		local bl = (b - 1) * seq_per_img + 1
		self.label_batch[{{bl, bl + seq_per_img -1}}] = seq

		------------------------------
		-- Keep some local info
		------------------------------
		local info_batch = {}
		info_batch.vid = self.info.video[b_ix].video_id	
		--info_batch.wrapped = wrapped -- this will be used in the evalution
		table.insert(meta_info, info_batch )
	end
	
	------------------------------
	-- Now make all batch one table and return it
	------------------------------
	local data = {}
	data.videos = self.vid_batch_raw
	data.labels = self.label_batch:transpose(1,2):contiguous() -- note: make label sequences go down as columns
	data.bounds = {it_pos_now = self.split_iter[split], it_max = #l_split_ix, wrapped = wrapped}
	data.info   = meta_info
	data.size   = batch_size --add this one, because for evaluation the batch_size can be adjusted depens on the input  

	return data
end	

function DataLoader:get_batch_frames_Conv(opt)
	--[[
           This function reads conv features and return a tensor 
	  ]]
	assert(self.i_type == 'vid_conv', string.format('Wrong input type ("%s"), should be:"vid_conv"',self.i_type ))  
 	local split = utils.get_kwarg(opt, 'split')
	local batch_size  = utils.get_kwarg(opt, 'batch_size', 5)
	local seq_per_img = utils.get_kwarg(opt, 'seq_per_img', 1) -- Keep this one for future compatibility 
	--local dtype = utils.get_kwarg(opt, 'dtype') -- Since doing mean here, lets do the mean in GPU


	-- check the current split is valid
	local l_split_ix = self.split_ix[split]
	assert(l_split_ix, string.format('The split "%s" is not found', split))

	local max_idx = #self.split_ix[split] -- len current split

    -- during the prediction, for each sample, we must only predict one
    -- so it is important to during test/eval, when we wrap around, adjust the batch size 
	if (split ~= 'train') then
		local diff = max_idx - self.split_iter[split] + 1
		if  diff < batch_size then
			batch_size = diff 
		end	
	end

	self.vid_batch_raw:resize(self.num_fm[1], batch_size, self.feat_size, self.conv_loc_size):zero() -- example: 16*128*512*196
	self.label_batch:resize(batch_size * seq_per_img, self.seq_length):zero()
	local wrapped = false
	local meta_info = {}

	for b = 1, batch_size do

		------------------------------
		-- Handle the split index
		------------------------------
		local c_idx =  self.split_iter[split] -- current index
		local next_idx = c_idx + 1

		if next_idx > max_idx then -- if the next_index is greater the length of current split
			next_idx =1
			wrapped = true
		end
		-- update iterator
		self.split_iter[split] = next_idx
		-- get current data
		local b_ix = l_split_ix[c_idx] 
		--some sanity check
		assert(b_ix ~= nil, 'Somthing wrong with the split')

		------------------------------
		-- Now fetch the videos from the input 
		-- Each row in the h5 is just one sample
		------------------------------
		local h5_row_key = self.i_type..tostring(b_ix)
		 -- The output is 16*512*14*14, we need to make them 16*512*196
		 -- 16*512*14*14 --> 16*512*196 
		self.vid_batch_raw[{{}, {b}, {}, {}}] = self.h5_file:read(h5_row_key):all():view(self.num_fm[1],self.feat_size, -1 )

		------------------------------
		-- Now fetch the captions: Not happy with the code, should be written simpler
		------------------------------
		local s_ix_start = self.label_start_ix[b_ix] -- index start for current input
		local s_ix_end   =  self.label_end_ix[b_ix]  -- index end for current input 
		local num_cap_ix = s_ix_end - s_ix_start + 1 -- number of captions available for this video

		if  num_cap_ix <= 0 then 
			error(string.format('No caption for current image "%d in batch %b"', b_ix, b))
		end
        
		local seq 
		if (num_cap_ix < seq_per_img) then
			-- no enough sample
			seq = torch.LongTensor(seq_per_img, self.seq_length)
			for q = 1, seq_per_img do
				local idx_label = torch.random(s_ix_start, s_ix_end )
				seq[{ {q,q} }] = self.h5_file:read('labels'):partial({idx_label, idx_label}, {1,self.seq_length})
			end
		else
			-- enough sample
			local idx_label = torch.random(s_ix_start, s_ix_end - seq_per_img + 1  )
			seq = self.h5_file:read('labels'):partial({idx_label, idx_label + seq_per_img - 1}, {1,self.seq_length})
		end

		local bl = (b - 1) * seq_per_img + 1
		self.label_batch[{{bl, bl + seq_per_img -1}}] = seq

		------------------------------
		-- Keep some local info
		------------------------------
		local info_batch = {}
		info_batch.vid = self.info.video[b_ix].video_id	
		--info_batch.wrapped = wrapped -- this will be used in the evalution
		table.insert(meta_info, info_batch )
	end
	
	------------------------------
	-- Now make all batch one table and return it
	------------------------------
	local data = {}
	data.videos = self.vid_batch_raw
	data.labels = self.label_batch:transpose(1,2):contiguous() -- note: make label sequences go down as columns
	data.bounds = {it_pos_now = self.split_iter[split], it_max = #l_split_ix, wrapped = wrapped}
	data.info   = meta_info
	data.size   = batch_size --add this one, because for evaluation the batch_size can be adjusted depens on the input  

	return data
end	

function DataLoader:clearState()
	self.vid_batch_raw:set()
	self.label_batch:set()
end

function DataLoader:get_rand_video()
	-- This function randmoly returns one video  
	local idx_vid = torch.random(1, self.num_vids )
	local one_vid = self.h5_file:read('vid_r'..tostring(idx_vid)):all() 
	return one_vid
end

function DataLoader:get_len_split(split)
	return #self.split_ix[split]
end
	
function DataLoader:reset_data_iter(split)
	-- this function resets the split to 1
	self.split_iter[split] = 1
end

function DataLoader:get_vocabSize()
	return self.vocab_size
end

function DataLoader:get_seqLength()
	return self.seq_length
end

function DataLoader:get_index_to_word()
	return self.ix_to_word
end

function DataLoader:get_VideoLength()
	-- return number of frames
	return self.num_fm[1]
end

function DataLoader:get_InputFeatSize()
	-- return the feat size
	return self.feat_size
end

function DataLoader:get_ConvMapSize()
	-- return convolution map size
	return self.conv_loc_size
end