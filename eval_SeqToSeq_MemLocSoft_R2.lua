require 'torch'
require 'nn'
require 'loadcaffe'

----------------------------------------------------
-----------------Load local modules-----------------
----------------------------------------------------
require 'misc.VideoToAvgPool'
local net_utils = require 'misc.net_utils'
local helper_utils = require 'misc.utils_helper'
require 'misc.DataLoader'
require 'models.SeqToSeqCriterion'

----------------------------------------------------
-----------------Options---------------------------
----------------------------------------------------
local cmd = torch.CmdLine()

-- Dataset options
cmd:option('-input_h5', '')
cmd:option('-json_file', '')

-- Video Feature options
cmd:option('-cnn_model','')
cmd:option('-cnn_proto', '')

-- Model options
cmd:option('-sample_max', 1)
cmd:option('-temperature', 1.0)
cmd:option('-beam_size', 5)
cmd:option('-batch_size', 5)
cmd:option('-split', 'test', 'which split to eval train|test|val')

-- Log options
cmd:option('-print_every', 100, 'How often to log the print out stuffs duirng train and validation')
cmd:option('-seed', 1234 , 'This is would be used to make the result reproducible')
cmd:option('-init_from', '', 'path to a model checkpoint to initialize model weights from.')
cmd:option('-lang_eval_method', 'CIDEr', 'BLEU/CIDEr/METEOR/ROUGE_L?')
cmd:option('-log_id', '_eval_ssl_')
cmd:option('-f_gt','', 'this file is used as ground truth')
cmd:option('-checkpoint_name', '')

-- Backend options
cmd:option('-gpu_id', -1)
cmd:option('-gpu_backend', 'nn','can be cuda|nn|cudnn')
cmd:text()

local opt = cmd:parse(arg)
opt.seq_per_img = 1 --just reset to 1 and keep it for future compatibility
----------------------------------------------------
-----------------S0:Set up GPU stuff----------------
----------------------------------------------------
-- Depend on the input option, this function set using 
-- GPU and related parameters
torch.manualSeed(opt.seed)
local dtype = 'torch.FloatTensor'
local gpu_params = helper_utils.set_GPU_config(opt)
gpu_params.seed_obj.manualSeed(opt.seed)
print("Random seed is set to", opt.seed)

----------------------------------------------------
---------------S0:Load and init the Netwrok Model --
----------------------------------------------------
local model = {}
local vocab 
if opt.init_from ~= '' then

	if string.find(opt.init_from, "nnet") then
		require 'models.SeqToSeqLocSoftMeMAtt_sNN_R2'
		print('Using SeqToSeqLocSoftMeMAtt_sNN_R2')

	else
		assert(false,"Not supported yet...")
	end 

	print('Initializing from ', opt.init_from)
	local checkpoint = torch.load(opt.init_from)
	model.lm = checkpoint.model.model_lm:clone() -- this only load a model with init/random weights in newest version
	for k, v in pairs(model) do
		v:type(gpu_params.dtype)
    end
    	
	local m_params = nil                         -- this update the model with the learned weights
	if ( checkpoint.model.params ~= nil) then    --this if provides backward compatibility 
		print("trained model is saved using newest version")
		m_params = model.lm:getParameters()    -- this update the model with the learned weights
		m_params:copy(checkpoint.model.params)       -- copy the learned weights   
	end

	--------------------------------------
	-- extract the parameters
	if opt.cnn_model == '' then
		opt.cnn_model =  checkpoint.opt.cnn_model
	end
	if opt.cnn_proto == '' then
		opt.cnn_proto =  checkpoint.opt.cnn_proto
	end

	opt.rnn_size = checkpoint.opt.rnn_size
	opt.input_encoding_size = checkpoint.opt.input_encoding_size  
	idx_to_word  = checkpoint.vocab
	opt.cnn_name = checkpoint.opt.cnn_name  
	opt.cnn_input_size =checkpoint.opt.cnn_input_size
	opt.tiny_save = checkpoint.opt.tiny_save 
	opt.video_length = checkpoint.opt.video_length 
	opt.input_type  = checkpoint.opt.input_type 
	opt.feat_input = checkpoint.opt.feat_input_size
	opt.conv_loc_size = checkpoint.opt.conv_loc_size
	opt.mem_size = checkpoint.opt.mem_size
	opt.mem_layers = checkpoint.opt.mem_layers
	opt.cnn_layer_name = checkpoint.opt.cnn_layer_name
	opt.layer_num = checkpoint.opt.layer_num
	opt.lang_eval_method = checkpoint.opt.lang_eval_method
	opt.embed_size = checkpoint.opt.embed_size

else
	assert(false,"This function can only be used for previously trained model")
end

if opt.video_length == nil then
	print('USED default value for opt.video_length')
	opt.video_length = 16 -- to support current version
end 
if opt.feat_input == nil then
	print('USED default value for opt.feat_input')
	opt.feat_input = 512  -- to support current version
end

print("Here are the parameters:")
print(opt)
-- add loss function and clones the model
--model.crit = nn.SeqToSeqCriterion(opt.video_length)
--model.lm:createClones() 
----------------------------------------------------
-- Update the types to GPU or CPU
print("Updating model parameters type to:\n", gpu_params.dtype)
for k, v in pairs(model) do
	v:type(gpu_params.dtype)
end

----------------------------------------------------
---------------S1:Process videos & extract feat-----
----------------------------------------------------
local video_process = VideoToAvgPool(opt)
-- This function exract features, calculates
--  means of features,
-- and pre-process the images 

opt.img_h5_path = video_process:compute_cnn_feat(opt, gpu_params)

----------------------------------------------------
---------------S2:Init data loader -----------------
----------------------------------------------------
local data_loader = DataLoader(opt)

print("\n")
print('*********************************************************************************')
print('********************')
print(string.format("The current reuslts will be reported on the split:'%s'", opt.split))
print('********************')
print('*********************************************************************************')  
print("\n")

----------------------------------------------------
-- evalution script
----------------------------------------------------
-- evalution script
local function eval_func(split)

	-- disable dropout
	model.lm:evaluate()
	data_loader:reset_data_iter(split)

	local loss_total =0
	local pred = {}
	local examples_seen = 0 
	local split_len = data_loader:get_len_split(split)
	local counter = 0
	local fgt = helper_utils.read_json(opt.f_gt)

	print("Start evalution...") 
	-- print some sanity check 
	print(string.format("split %s, size %d", split, data_loader:get_len_split(split))) 

	while true do

		-- get the data
		local m_b = data_loader:get_batch_frames_Conv({batch_size = opt.batch_size, split = split})
		m_b.videos = m_b.videos:type(gpu_params.dtype)
		m_b.labels = m_b.labels:type(gpu_params.dtype)
		examples_seen = examples_seen + m_b.size
		counter = counter + 1
       
		------------------------------------------------
		-- Forward pass
		local x, y = m_b.videos, m_b.labels

		local lm_data_eval   = {x, y}
		--local scores_eval    = model.lm:forward(lm_data_eval)
		--local loss_curr      = model.crit:forward(scores_eval, y)     
		--loss_total = loss_total + loss_curr

		-- generate samples/sentences and keep some infors
		local seq, _ = model.lm:sample(x, opt) 
		local sents = net_utils.decode_sequence(idx_to_word, seq)

		for s = 1, #sents do
			--if the split is wrapped, we need to exit the loop, one example can not be added two times
			-- this is very important especially in the evalution score
			--if m_b.info[s].wrapped == true then
			--  break
			--end
			table.insert(pred, {video_id = m_b.info[s].vid, caption = sents[s]})
			if opt.print_every > 0 and counter % opt.print_every == 0  then
				vid_id = m_b.info[s].vid
				print(string.format("Caption '%s' for image '%s with GT: %s'", sents[s], m_b.info[s].vid, fgt[ vid_id ][1].caption))
			end
		end

		if m_b.bounds.wrapped or examples_seen >= split_len then
			break
		end
          
	end

	-- evaluate the language model 
	local lang_stats
	if opt.lang_eval_method ~= '' then
		lang_stats = net_utils.language_eval(opt, pred)
	end
    
	collectgarbage() -- just to do some performance stuff
	return loss_total / counter, pred, lang_stats
end

local loss, preds, lang_stats = eval_func(opt.split)
if lang_stats then
	print("Lang stats:", lang_stats)
end 
print(string.format("loss for '%s' is: %f", opt.split, loss))
