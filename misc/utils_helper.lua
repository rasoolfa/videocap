local cjson = require 'cjson'
local utils_helper ={}

function utils_helper.get_kwarg(kwargs, key, default_value)
	--[[
		This function gets the options dict and returns its 
		value based on the input key
	]]
	if default_value == nil and (kwargs == nil or kwargs[key] == nil) then
		assert(false, string.format('"%s" is required and was not given', key))
	end
	if kwargs[key] == nil then
		return default_value
	else
		return kwargs[key]
	end
end

function utils_helper.set_GPU_config(opt, input_type)
    --[[
        This function set the gpu and related configurations
      ]]
   	local gpu_backend   = utils_helper.get_kwarg(opt, 'gpu_backend', 'nn')
	local gpu_id  		= utils_helper.get_kwarg(opt, 'gpu_id', 0) -- -1 use cpu
   	local dtype = 'torch.FloatTensor'
   	local seed_obj = torch

   	if input_type == 'cnn' then
   		print("********************************************")
   		print(" Some of loadcaffe modules only support cudnn/nn/ccn2, refer to https://github.com/szagoruyko/loadcaffe")
   		print("********************************************")
   	end
   	if (gpu_backend == 'nn' and gpu_id >= 0) or (gpu_backend == 'cuda' and gpu_id == -1) then
   		print("To use gpu, the 'gpu_id >= 0' AND 'gpu_backend' should be set to 'cuda|cudnn' not 'nn'")
   	end

	gpu_params ={}
	gpu_params['gpu_backend'] = gpu_backend

	if gpu_backend == 'nn' or gpu_id == -1 then
		require 'nn'
		print('Running on CPU')
		gpu_obj = nn
	elseif gpu_backend == 'cuda' and gpu_id >= 0 then  -- some of loadcaffe only supports cudnn/nn/ccn2
		require 'cutorch'
		require 'cunn'
		cutorch.setDevice(gpu_id + 1)
		dtype = 'torch.CudaTensor'
		gpu_obj = nn  -- this is only used for loadCaffee
		print(string.format('Running with cuda on GPU %d', gpu_id))
		seed_obj = cutorch
	elseif gpu_backend == 'cudnn' then
		require 'cutorch'
		require 'cunn'
		require 'cudnn'
		dtype = 'torch.CudaTensor'
		cutorch.setDevice(gpu_id + 1)
		print(string.format('Running with cudnn on GPU %d', gpu_id))
		gpu_obj = cudnn
		seed_obj = cutorch

	else
		assert(false, string.format('Unsupported gpu_backend "%s"', gpu_backend))
	end

	gpu_params['gpu_obj'] = gpu_obj
	gpu_params['dtype'] = dtype
	gpu_params['seed_obj'] = seed_obj

	return gpu_params	
end

function utils_helper.reset_type()
	return 'torch.FloatTensor'
end

function utils_helper.read_json(f_name)
	local i_file = io.open(f_name, 'r')
	local data =  i_file:read('*all')
	i_file:close()
	return cjson.decode(data)
end

function utils_helper.write_json(f_name, data)
	local json_obj = cjson.encode(data)
	local out_file = io.open(f_name, 'w')
	out_file:write(json_obj)
	out_file:close()
end

function utils_helper.get_dict_len(i_dict)

	local counter = 0
	for k,v in pairs(i_dict) do
		counter = counter + 1
	end

	return counter
end

function utils_helper.file_exits(file_name)
	--[[
	   check if the file exist 
	   ]]	
  	local file_found=io.open(file_name, "r")  
  	if file_found == nil then
        return false
  	else
        return true
    end
end

function utils_helper.create_newFileName(file_name, new_name)
	local path, name, _ = string.match(file_name, "(.-)([^//]-([^%.]+))$")
	local new_fname = path..new_name..name

	return new_fname
end
function utils_helper.create_newFileName_v2(file_name, new_name, cnn_name)
	local path, name_w_ext, ext_f = string.match(file_name, "(.-)([^//]-([^%.]+))$")
	name   = string.gsub(name_w_ext ,'.h5', '')
	c_name = string.lower(string.gsub(cnn_name, '-',''))
	local new_fname = path..new_name..name ..'_'..c_name..'.'..ext_f

	return new_fname
end
function utils_helper.print_separator()
	-- Funny functions to draw lines 
	print("--------------------------------------")
end

return utils_helper
