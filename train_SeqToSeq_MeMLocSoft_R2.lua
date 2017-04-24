require 'torch'
require 'nn'
require 'loadcaffe'

--[[
Train a network for vidoe understanding
	o. Build a data loader 
	o. Build a language model 
	o. Extract features
	o. Train model
	o. Generate desciption
]]

----------------------------------------------------
-----------------Load local modules-----------------
----------------------------------------------------
require 'misc.VideoToAvgPool'
local net_utils = require 'misc.net_utils'
local helper_utils = require 'misc.utils_helper'
require 'misc.DataLoader'
require 'models.SeqToSeqCriterion'
require 'misc.optim_updates'

----------------------------------------------------
-----------------Options---------------------------
----------------------------------------------------
local cmd = torch.CmdLine()

-- Dataset options
cmd:option('-input_h5', '')
cmd:option('-json_file', '')
cmd:option('-dname', ‘yt’, 'dataset name that will be used here')

-- Video Feature options
cmd:option('-feat_method', 'caffe_pool')
cmd:option('-cnn_model','')
cmd:option('-cnn_proto', '')

cmd:option('-cnn_name', 'VGG-16')   
cmd:option('-cnn_layer_name', 'relu5_4') 
cmd:option('-layer_num', 42)   

cmd:option('-cnn_input_size', 224) --All those pre-trained nets accepts 224, to be easily extensible to new network
cmd:option('-input_encoding_size', 512, 'to map from video features to LSTM state')    --Encoding for weight between video and LSTM
--cmd:option('-train_en_layer', '1')    --Encoding for weight between video and LSTM
--cmd:option('-tiny_save', 2, 'if 1 means only avg features will be saved in dataprepation step, 2 means cnn FC7 features without avg')

-- Optimization options
cmd:option('-batch_size', 16)
cmd:option('-dropout', 0.5,'Dropout rate for RNN to apply over non-recurrent weights')
cmd:option('-grad_clip', 2,'clip gradients at this value') -- to overcome exploding gradient, clamp the grads 
cmd:option('-weight_decay', 0, 'L2 just for the encoding layer') -- to overcome exploding gradient, clamp the grads 
cmd:option('-lr_decay_every', 200, 'to decay learning rate')
cmd:option('-lr_decay_factor', 0.1, 'learning rate decay factor')
cmd:option('-max_epoch', 400, 'max number of epoch')
cmd:option('-learning_rate', 2e-05, 'learning_rate')
cmd:option('-optim','adam','what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
cmd:option('-optim_alpha',0.8,'alpha for adagrad/rmsprop/momentum/adam')
cmd:option('-adam_beta1’,0.8,’beta used for adam')
cmd:option('-adam_beta2',0.999,'beta used for adam')
cmd:option('-optim_epsilon',1e-8,'epsilon that goes into denominator for smoothing')
cmd:option('-mem_size', 797, 'number of hidden nodes in Memory Network')
cmd:option('-mem_layers', 1, 'number of Memory layers')
cmd:option('-lr_cap',1e-10,'learning rate cap')
cmd:option('-weightDecayOptim',1e-05, 'weight Decay for L2 inside optim modules')

-- Model options
cmd:option('-model_type', 'lstm')
cmd:option('-rnn_size', 1479,’Number of hidden nodes in RNN in each layer')
cmd:option('-num_layers', 2,'Number of RNN layers')
cmd:option('-sample_max', 1)
cmd:option('-temperature', 1.0)
cmd:option('-beam_size', 1)
cmd:option('-init_type_loc', 'nnet', 'Init the h0 and c0 with zero or nnet')
cmd:option('-embed_size', 402, 'this is word embedding size')

-- Log options
cmd:option('-print_every', 100, 'How often to log the print out stuffs duirng train and validation')
cmd:option('-seed', 1234 , 'This is would be used to make the result reproducible')
cmd:option('-checkpoint_every', 600, 'how often to save a model checkpoint?')
cmd:option('-checkpoint_name', 'cv/checkpoint')
cmd:option('-init_from', '', 'path to a model checkpoint to initialize model weights from. Empty = don\'t')
cmd:option('-lang_eval_method', 'METEOR', 'BLEU/CIDEr/METEOR/ROUGE_L?')
cmd:option('-losses_log_every', 25, 'How often to log the loss value (0 = disable)')
cmd:option('-log_id', '0', 'an id identifying this run/job. used in cross-val and appended when writing progress files')
cmd:option('-speed_benchmark', 1, 'use to time the forward/backward execution')
cmd:option('-f_gt','', 'this file is used as ground truth')


-- Backend options
cmd:option('-gpu_id', 0)
cmd:option('-gpu_backend', ‘cuda’,’can be cuda|nn|cudnn')
cmd:text()

local opt = cmd:parse(arg)
opt.seq_per_img = 1 --just reset to 1 and keep it for future compatibility
-- since this one expecting video frames, we manualy set this, and should be replaced with following code  
-- cmd:option('-tiny_save', 2, 'if 1 means only avg features will be saved in dataprepation step, 2 means cnn FC7 features without avg')
opt.tiny_save = 3  
opt.input_type = 'vid_conv' -- this should be changed, just show the which dataset to grab 
--opt.cnn_layer_name = 'conv5_3'

----------------------------------------------------
-----------------S0:Set up GPU stuff----------------
----------------------------------------------------
-- Depend on the input option, this function set using 
-- GPU and related parameters
torch.manualSeed(opt.seed)
local dtype = 'torch.FloatTensor'
local gpu_params = helper_utils.set_GPU_config(opt)
gpu_params.seed_obj.manualSeed(opt.seed)
opt.dtype = gpu_params.dtype --will be used in dataloader
print("Random seed is set to", opt.seed)

----------------------------------------------------
---------------S1:Process videos & extract feat-----
----------------------------------------------------
local video_process = VideoToAvgPool(opt)

-- This function exract features, calculates
--  means of features,
-- and pre-process the images 
opt.img_h5_path = video_process:compute_cnn_feat(opt, gpu_params)
video_process = nil -- no longer required

----------------------------------------------------
---------------S2:Init data loader -----------------
----------------------------------------------------
local data_loader = DataLoader(opt)
local vid_length = data_loader:get_VideoLength() -- return number of frames/video
opt.conv_loc_size = data_loader:get_ConvMapSize()
local feat_input_size = data_loader:get_InputFeatSize()
opt.video_length = vid_length
opt.feat_input = feat_input_size
----------------------------------------------------
---------------S2:Init the Netwrok Model -----------
----------------------------------------------------

-- Steps for this part
-- Check if we should start from scratch or some saved-model
-- Update model types cuda or non-cuda parts
-- Flatten all parameters 

local model = {}
local lm_config = {}

function setup_LM()
	--Params init
	lm_config.rnn_size = opt.rnn_size
	lm_config.num_layers = opt.num_layers
	lm_config.dropout = opt.dropout
	lm_config.seq_length = data_loader:get_seqLength()
	lm_config.vocab_size = data_loader:get_vocabSize()
	lm_config.sample_max = opt.sample_max
	lm_config.temperature = opt.temperature
	lm_config.beam_size = opt.beam_size
	lm_config.input_encoding_size = opt.input_encoding_size
	lm_config.batch_size = opt.batch_size
	lm_config.video_length = vid_length
	lm_config.feat_input = feat_input_size
	lm_config.conv_loc_size = opt.conv_loc_size
	lm_config.mem_size = opt.mem_size
	lm_config.mem_layers = opt.mem_layers
	lm_config.embed_size = opt.embed_size


	--Model init
	if opt.init_type_loc == 'zero' then 
		require 'models.SeqToSeqLocSoftMeMAtt_R2'
		model.lm = nn.SeqToSeqLocSoftMeMAttModel(lm_config)
		print('Using SeqToSeqLocSoftMeMAtt_R2 ...')

	elseif 	opt.init_type_loc == 'nnet' then
		require 'models.SeqToSeqLocSoftMeMAtt_sNN_R2'
		model.lm = nn.SeqToSeqLocSoftMeMAtt_sNN_R2(lm_config)
		print('Using SeqToSeqLocSoftMeMAtt_sNN_R2 ...')

	else
		  assert(false,string.format("init_type %s not supported yet...", opt.init_type))

	end 

	--Model init
	--model.lm = nn.SeqToSeqLocSoftMeMAttModel(lm_config)
	model.crit = nn.SeqToSeqCriterion(lm_config.video_length)

end

if opt.init_from ~= '' then
	print('Initializing from ', opt.init_from)
	assert(false,"Not supported yet...")
	local checkpoint = torch.load(opt.init_from)
	-- load previous model 
else
	setup_LM()
	print("Input parameters to the LM:", lm_config)
end

----------------------------------------------------
-- Save the model with init values, this thin_model model will not be updated
-- One advantage of having modellike this to avoid saving the clones and grad values
local thin_model_lm = model.lm:clone()

----------------------------------------------------
-- Update the types to GPU or CPU
print("Updating model parameters type to:\n", gpu_params.dtype)
for k, v in pairs(model) do
	v:type(gpu_params.dtype)
end
--torch.setdefaulttensortype(gpu_params.dtype) not a good idea - few api works with cutroch

helper_utils.print_separator()

----------------------------------------------------
-- Flatten all parameters and make ready params 
local params_lm, grad_params_lm = model.lm:getParameters()
print(" Total parameters of SeqToSeqLocSoftMeMAtt_sNN_R2:", params_lm:nElement())
-- some sanity checks
assert(params_lm:nElement() ==  grad_params_lm:nElement() , 'Size of grad params and params must be the same for LM, sth is wrong')
helper_utils.print_separator()

model.lm:createClones()
collectgarbage() 
----------------------------------------------------
---------------S3:Setup loss & eval functions ------
----------------------------------------------------
local forward_backward_times = {}

----------------------------------------------------
-- loss function 
local function loss_func(p_for_back)

	-- Add dropout only during training
  	model.lm:training()

  	--zero out the parameters
  	grad_params_lm:zero()

  	--get data
  	opt.split = 'train'
  	local mini_b_data = data_loader:get_batch_frames_Conv(opt)
  	mini_b_data.videos = mini_b_data.videos:type(gpu_params.dtype)
  	mini_b_data.labels = mini_b_data.labels:type(gpu_params.dtype)

  	-- time the forward/backward
  	-- Start the timer 
  	local timer
  	if opt.speed_benchmark == 1 then
		if gpu_params.gpu_obj == cutorch then 
			cutorch.synchronize() 
		end
		timer = torch.Timer()
  	end

	------------------------------------------------
  	-- Forward pass
  	local lm_data   = {mini_b_data.videos, mini_b_data.labels}
  	local scores    = model.lm:forward(lm_data)
  	local loss      = model.crit:forward(scores, mini_b_data.labels)

	------------------------------------------------
	-- Backward Pass
	-- Run the Criterion and model backward to compute gradients
	local  grad_scores = model.crit:backward(scores, mini_b_data.labels )
	local  grad_lm     = model.lm:backward(lm_data, grad_scores)
	local  grad_vid, _ = unpack(grad_lm) 
	------------------------------------------------
	-- Grad clipping and apply L2 
	if opt.grad_clip > 0 then
		--LM layer
		grad_params_lm:clamp(-opt.grad_clip, opt.grad_clip)
	end

  	------------------------------------------------
	-- keep track of how long the forward and backward take
	if timer then
		if gpu_params.gpu_obj == cutorch then 
			cutorch.synchronize() 
		end
		local time = timer:time().real
		if p_for_back == 1 then
			print('Forward and Backward passes took ', time)	
		end
		table.insert(forward_backward_times, time)
  	end

	model.lm:clearState()
	data_loader:clearState()

	return loss
  
end

----------------------------------------------------
-- evalution script
local function eval_func(split)

	-- disable dropout
	model.lm:evaluate()
	data_loader:reset_data_iter(split)

	local loss_total = 0
	local pred = {}
	local examples_seen = 0 
	local split_len = data_loader:get_len_split(split)
	local idx_to_word =data_loader:get_index_to_word()
	local counter = 0
	print("Start evalution...") 

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
  		local scores_eval    = model.lm:forward(lm_data_eval)
  		local loss_curr      = model.crit:forward(scores_eval, y)	  	
  		loss_total = loss_total + loss_curr

  		-- generate samples/sentences and keep some infors
  		local seq, _ = model.lm:sample(x, opt) 
  		local sents = net_utils.decode_sequence(idx_to_word, seq)

  		for s = 1, #sents do
  			--if the split is wrapped, we need to exit the loop, one example can not be added two times
  			-- this is very important especially in the evalution score
			table.insert(pred, {video_id = m_b.info[s].vid, caption = sents[s]})
			if opt.print_every > 0 and counter % opt.print_every == 0  then
				print(string.format("Caption '%s' for image '%s'", sents[s], m_b.info[s].vid))
			end
  		end

  		if m_b.bounds.wrapped or examples_seen >= split_len then
  			break
  		end
		data_loader:clearState()
          
	end

	-- evaluate the language model 
	local lang_stats
	if opt.lang_eval_method ~= '' then
		lang_stats = net_utils.language_eval(opt, pred)
	end
   	
   	collectgarbage() -- just to do some performance stuff
	return loss_total / counter, pred, lang_stats
end

----------------------------------------------------
---------------S3: Main loop -----------------------
----------------------------------------------------
-- 
local optim_config = {}
local optim_state_lm = {}
local train_loss_hist = {}
local val_lang_stats_hist = {}
local val_loss_hist = {}
local best
local loss_0
local num_train = data_loader:get_len_split('train')
local num_iterations = opt.max_epoch * num_train
local print_for_back_tim = 0
local iter_per_epoch =  math.floor(num_train / opt.batch_size) 
opt.learning_rate_init = opt.learning_rate -- since changing lr, just keep the init value 
local c_iter_decay_fact = 0
for iter = 0, num_iterations do

	------------------------------------------------
	-- forward/backward for the current the mini batch
	local loss_1 = loss_func(print_for_back_tim) 
	print_for_back_tim = 0
	------------------------------------------------
	-- Some prinitng 
	local epoch = math.floor((iter * opt.batch_size) / num_train) + 1

	if opt.print_every > 0 and iter % opt.print_every == 0 then
		table.insert(train_loss_hist, loss_1) 
		local float_epoch = (iter * opt.batch_size)  / num_train + 1
		local msg = 'Epoch %.2f / %d, iter = %d / %d, loss = %f'
		local args = {msg, float_epoch, opt.max_epoch, iter, num_iterations, loss_1}
		print(string.format(unpack(args)))
		print_for_back_tim = 1
	end

	------------------------------------------------
	-- do validation once a while and save that as checkpoints
	-- including final iteration 
	if (iter > 0 and opt.checkpoint_every > 0) and (iter % opt.checkpoint_every == 0 or iter == num_iterations) then
		------------------------------------------------
		-- evaluate the validation performance
		local val_loss, val_preds, lang_stats = eval_func('val')

		------------------------------------------------
		-- First write data into thin json file
		table.insert(val_lang_stats_hist, lang_stats)
		table.insert(val_loss_hist, val_loss)

		if lang_stats ==nil then
			print(string.format("Valdiation loss %f ",val_loss, lang_stats))
		else
			print(string.format("Valdiation loss %f ",val_loss)," lang_stats:", lang_stats)
		end
		
		local checkpoint = {
			opt = opt,
			train_loss_hist = train_loss_hist,
			val_loss_hist = val_loss_hist,
			forward_backward_times = forward_backward_times,
			val_lang_stats_hist = val_lang_stats_hist,
			val_preds = val_preds,
			epoch = epoch,
			iter = iter}

		local filename = path.join(opt.checkpoint_name, string.format('%s.json', opt.log_id))
		-- Make sure the checkpoint dir exists before try to write it
		paths.mkdir(paths.dirname(filename))
		helper_utils.write_json(filename, checkpoint)
		print(string.format("json Checkpoint is written for iter %d in %s:",iter, filename))

		-- Now it is time to save the model
		local curr_score, curr_score_alt
		if lang_stats ~= nil  then
			curr_score = lang_stats[opt.lang_eval_method]
			if opt.lang_eval_method == 'METEOR' then
				curr_score_alt = lang_stats['Bleu_4']
			else
				curr_score_alt = lang_stats['METEOR']
			end		
		else
			-- if no language score just use -loss
			curr_score = -val_loss
		end
		if best == nil or curr_score > best.score or ( torch.abs(curr_score - best.score) < 0.00001 and curr_score_alt > best.score_alt )then

			-----------------------------------------------------
			-- Save best model and other stuffs
			best = {} 
			best.score = curr_score
			best.score_alt = curr_score_alt
			best.loss =  val_loss
			checkpoint.score = best.score
			checkpoint.score_alt = best.score_alt
			checkpoint.best_loss = best.loss

			local filename_best = path.join(opt.checkpoint_name, string.format('%s_%d.json', opt.log_id, 0))
			-- Make sure the checkpoint dir exists before try to write it
			paths.mkdir(paths.dirname(filename_best))
			helper_utils.write_json(filename_best, checkpoint)
			print(string.format("Best json Checkpoint is written for iter %d in %s:",iter, filename_best))

			local save_model = {}
			save_model.model_lm = thin_model_lm
			save_model.params = params_lm:clone() -- this one will be used to load and init the new model and it is very small 
			checkpoint.model =  save_model
			checkpoint.vocab = data_loader:get_index_to_word()
			local filename_model = path.join(opt.checkpoint_name, string.format('%s.t7', opt.log_id))
			-- Make sure the checkpoint dir exists before try to write it
			paths.mkdir(paths.dirname(filename_model))
			torch.save(filename_model, checkpoint)
			print(string.format("model Checkpoint is written for iter %d in %s:",iter, filename_model))
		end

	end

	------------------------------------------------
	-- Learning rate decay
	------------------------------------------------
	-- decay the learning rate 
	if opt.lr_decay_every > 0 and iter > 0 and iter % (iter_per_epoch * opt.lr_decay_every) == 0 then
		c_iter_decay_fact = c_iter_decay_fact + 1
		local temp_lr  = opt.learning_rate / (1.0 + c_iter_decay_fact * 1e-3)
		if temp_lr > opt.lr_cap then -- don't let to decrease dramatically 
		    opt.learning_rate = temp_lr -- set the decayed rate
		    print(string.format('Update lr to %s in iter %d with iter_per_epoch %d', opt.learning_rate, iter, iter_per_epoch))
		end
	end
	-- Parameters update 
	if opt.optim == 'adam' then
		optim_config.learningRate = opt.learning_rate
		optim_config.beta1 = opt.adam_beta1
		optim_config.beta2 = opt.adam_beta2
		optim_config.epsilon = opt.optim_epsilon
		optim_config.weightDecay = opt.weightDecayOptim
		adam(params_lm, grad_params_lm, optim_config, optim_state_lm)
	elseif opt.optim == 'rmsprop' then
		optim_config.learningRate = opt.learning_rate
		optim_config.alpha = opt.optim_alpha
		optim_config.epsilon = opt.epsilon	
		rmsprop(params_lm, grad_params_lm, optim_config, optim_state_lm)
	else
		assert(false, string.format(" '%s' is not supported yet as an optim", opt.optim))
	end

	------------------------------------------------
	-- some prinitng and other stuffss
	if iter == 0 then
		loss_0 = loss_1
	end

	------------------------------------------------
	-- Stopping criteria
	if loss_1 > loss_0 * 20.0 then
		print("Loss is increasing, exiting ...")
		break
	end
   	collectgarbage() -- just to do some performance stuff

end

----Run the on the test data 
helper_utils.print_separator()
print(string.format("Here is the best results for '%s' is", opt.split))
if best then
	print("The loss is",best.loss)
	print("The best score is",best.score)	
	print(best.score)
end

