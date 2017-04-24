require 'loadcaffe'

local net_utils = {}
local utils = require 'misc.utils_helper'


function net_utils.build_cnn(opt, gpu_params)
	--[[ This function take the caffe CNN and build one in the torch
	     It support all VGG architectures, i.e. VGG-16 and VGG-19
	     -- VGG_16: 38 means extract from fully connected layer(fc7)
	     -- VGG_19: 42 means extract from fully connected layer(fc7)
    	]]
	local layer_num = utils.get_kwarg(opt, 'layer_num', 38) --remeber these 38 layers contain dropout, remeber to turn it off
	local cnn_proto = utils.get_kwarg(opt, 'cnn_proto')
	local cnn_model = utils.get_kwarg(opt, 'cnn_model')
	local cnn_name = utils.get_kwarg(opt, 'cnn_name')

	local cnn_backend = utils.get_kwarg(gpu_params, 'gpu_backend')
	local gpu_obj = utils.get_kwarg(gpu_params, 'gpu_obj')
	local dtype 	= utils.get_kwarg(gpu_params, 'dtype')

	-- add this option to extract specific layer
	local layer_name 	= utils.get_kwarg(opt, 'cnn_layer_name', "")
	-- now load the caffee 
	local cnn = loadcaffe.load(cnn_proto, cnn_model, gpu_backend)

	-- now build the cnn
	-- remeber caffe inputs by default are BGR, since we are using opencv, we meet this requirment
	local cnn_net = nn.Sequential()

	for i = 1, layer_num do

		local layer_id = cnn:get(i)
		cnn_net:add(layer_id)
        -- add this line to stop at a given layer 
		if opt.cnn_layer_name ~= "" and layer_id.name == opt.cnn_layer_name then
			break
		end

	end

	--make sure they have corret type
	cnn_net:type(dtype)
	cnn:type(dtype)
	return cnn, cnn_net
end

function net_utils.build_cnn_inception(opt, gpu_params)

	--[[ 
		This function take the inception v3 and return a network for feature extraction.
		Right now it is very specif to inception v3
	 ]]
	local layer_num = utils.get_kwarg(opt, 'layer_num', 30) --remeber these 30
	local cnn_model = utils.get_kwarg(opt, 'cnn_model')
	local cnn_name = utils.get_kwarg(opt, 'cnn_name')
	local cnn_backend = utils.get_kwarg(gpu_params, 'gpu_backend')
	local dtype 	= utils.get_kwarg(gpu_params, 'dtype')
	-- now load the model 
	local cnn = torch.load(cnn_model)

	-- now build the cnn
	-- remeber caffe inputs by default are BGR, since we are using opencv, we meet this requirment

	local cnn_net = nn.Sequential()
	for i = 1, layer_num do
		local layer_id = cnn:get(i)
		cnn_net:add(layer_id)

	end
	
	cnn_net:type(dtype)

	return cnn_net
end
function net_utils.preprocess_imgs(imgs, gpu_params, opt)
	--[[
	Preporcess the data to be compatible with CNN network
	Right now, it is specific to VGG-style networks, should be adopted to future network
	]]

	local h, w = imgs:size(3), imgs:size(4)
	local cnn_input_size = utils.get_kwarg(opt, 'cnn_input_size')

	-- simply crop the input image from center 
	if h > cnn_input_size or w > cnn_input_size then
		local xoff, yoff
		xoff, yoff = math.ceil((w - cnn_input_size) / 2),math.ceil((h - cnn_input_size) / 2)
		imgs = imgs[{ {}, {}, {yoff, yoff + cnn_input_size - 1}, {xoff, xoff + cnn_input_size - 1} }]
	end

	-- The images are in byte, need to convert to the current type: float or cuda
	imgs = imgs:type(gpu_params.dtype)

	if not net_utils.vgg_mean then
		--net_utils.vgg_mean = torch.FloatTensor{123.68, 116.779, 103.939}:view(1,3,1,1) -- in RGB order
		net_utils.vgg_mean = torch.FloatTensor{103.939, 116.779, 123.68}:view(1,3,1,1) -- in BGR order, remember our input are in BGR 
	end
	net_utils.vgg_mean = net_utils.vgg_mean:type(gpu_params.dtype)

	--subtract vgg means
	imgs:add(-1, net_utils.vgg_mean:expandAs(imgs))

	return imgs
end

function net_utils.linear_encoding(opt)
	local feat_input =  utils.get_kwarg(opt, 'feat_input', 4096)
  	local enc_size =  utils.get_kwarg(opt, 'input_encoding_size', 512)
  	local vid_len  =  utils.get_kwarg(opt, 'video_length', 1) --for longer should user linear-embedding layer
    
    -- Create one layer feat_input*enc_size layer 
    local enco_layer = nn.Sequential()
	enco_layer:add(nn.Linear(feat_input, enc_size))
	enco_layer:add(nn.ReLU(true)) -- true = in-place, false = keeping separate state.
	return enco_layer

 end

function net_utils.clone_list(lst)
  -- takes list of tensors, clone all
  local new = {}
  for k,v in pairs(lst) do
    new[k] = v:clone()
  end
  return new
end

function net_utils.decode_sequence(ix_to_word, seq)
	    -- Given the seqt decodes to a sentence
	    -- ix_to_word: dict that maps a word index to a word
	    -- seq: is a LongTensor of size DxN with elements 1..vocab_size+1 (where last dimension is END token)  

  local D, N = seq:size(1), seq:size(2)
  local decoding_res = {}
  for i = 1, N do
    local sent = ''
    for j = 1, D do
      local ix = seq[{j, i}]
      local decoded_word = ix_to_word[tostring(ix)]

      if not decoded_word then 
      	break -- END token, likely. Or null token 
      end 
      if j >= 2 then -- add space between words
      	sent = sent .. ' ' 
      end
      sent = sent .. decoded_word
    end
    table.insert(decoding_res, sent)
  end

  return decoding_res
end

function net_utils.get_gModule_info(input_gModule)
	-- This function returns information about the module 
	local type_layer = {}
	local weights_dim = {}
	local layer = {}
	for indexNode, node in ipairs(input_gModule.forwardnodes) do
    	if node.data.module then
        	table.insert(type_layer, torch.type(node.data.module))
        	table.insert(layer, node.data.module)
        	if node.data.module.weight then
        		table.insert(weights_dim, node.data.module.weight:size())
        	end
        end	
    end
    
    local info ={}
    info.type = type_layer
    info.dims = weights_dim
    info.modules = layer

    return info
 
end
---------------------------------------------------------------------------------------------
--  These following functions idea/codes from https://github.com/karpathy/neuraltalk2/blob/master/misc/net_utils.lua
---------------------------------------------------------------------------------------------
function net_utils.listModules(input_net)
	--- Here we get the list of modulues
	local t = torch.type(input_net)
    local module_list 
	if t == 'nn.gModule' then
		module_list = net_utils.get_gModule_info(input_net).modules
	else
		module_list = input_net:listModules()
	end
	return module_list
end

function net_utils.sanitize_gradients(input_net)
	--- The only job this module does to remove the
	--- gradients so when we save the model, it takes
	--- less sapce
	local m_list =  net_utils.listModules(input_net)

	for k, v in ipairs(m_list) do
		if v.weight and v.gradWeight then
			v.gradWeight = nil
		end
		if v.bias and v.gradBias then
			v.gradBias = nil
		end
	end
end
function net_utils.unsanitize_gradients(input_net)
	--- The only job this module does to zero out the
	--- gradients so when we save the model, it takes
	--- less sapce
	local m_list =  net_utils.listModules(input_net)

	for k, v in ipairs(m_list) do
		if v.weight and (not v.gradWeight) then
			v.gradWeight = v.weight:clone():zero()
		end
		if v.bias and (not v.gradBias) then
			v.gradBias = v.bias:clone():zero()
		end
	end
end


function net_utils.language_eval(opt, preds)
  -- Save result in json, call python to evaluate and save result in
  -- json again. 
   local f_gt    = utils.get_kwarg(opt, 'f_gt')
   local log_id  = utils.get_kwarg(opt, 'log_id')
   local split =  utils.get_kwarg(opt, 'split')
   local ck_name = utils.get_kwarg(opt, 'checkpoint_name')
   local save_dest = ck_name .. '/score_js_' .. split 
   
   paths.mkdir(save_dest) 
   local out_struct = {v_preds = preds}
   local ran_id = torch.random(8000)
   local f_res = save_dest .. '/v' .. log_id .. '_' .. ran_id .. '.json' 
   
   utils.write_json(f_res, out_struct)
   local command = 'python eval_caption/scores_captions.py --f_gt ' .. f_gt .. ' --f_res ' .. f_res
   print(string.format("Executing %s",command))
   os.execute(command)
   
   local output = save_dest ..'/cv' .. log_id .. '_' .. ran_id .. '.json' 
   print(string.format("Result in %s",output))
   local result_struct = utils.read_json(output) 
   return result_struct

end

return net_utils









		









