require 'nn'
local utils = require 'misc.utils_helper'
local net_utils = require 'misc.net_utils'
local LSTM_3D = require 'component.LSTM_3D'
local LSTM    = require 'component.LSTM'
local INIT_STATE_NN = require 'component.InitStateNN' 
local SOFT_ATT = require 'component.SoftAttention_3D'
require 'component.LocAttention'

-------------------------------------------------------------------------------
-- Some notes
-------------------------------------------------------------------------------
--[[
     LSTM module create a an LSTM unit with multiple layers not states, basically 
     it creates a vertical units
     _createInitState: create init_state varibales to keep track of one layer 
     clone, create LSTM unit through times 
]]

-------------------------------------------------------------------------------
-- Seq2Seq with Location and Soft attention Model with Memory 
-------------------------------------------------------------------------------

local layer, parent = torch.class('nn.SeqToSeqLocSoftMeMAtt_sNN_R2', 'nn.Module')
function layer:__init(opt)
  parent.__init(self)

  -- options for core network
  self.vocab_size = utils.get_kwarg(opt, 'vocab_size') -- required
  self.input_encoding_size = utils.get_kwarg(opt, 'input_encoding_size')
  self.rnn_size = utils.get_kwarg(opt, 'rnn_size')
  self.num_layers = utils.get_kwarg(opt, 'num_layers', 1)
  self.conv_loc_size =  utils.get_kwarg(opt, 'conv_loc_size', 512)
  local dropout = utils.get_kwarg(opt, 'dropout', 0)
  self.embed_size = utils.get_kwarg(opt, 'embed_size', 256) 

  -- options for the Models
  self.seq_length = utils.get_kwarg(opt, 'seq_length')
  self.video_length = utils.get_kwarg(opt, 'video_length')
  self.total_seq_len = self.video_length + self.seq_length + 1 -- this includes start and end frames

  -- option for memory
  self.mem_size = utils.get_kwarg(opt, 'mem_size', self.rnn_size)
  self.mem_layers = utils.get_kwarg(opt, 'mem_layers', 1)
  self.mem_input_size = self.rnn_size  -- memory inputs are from our LSTM model
  self.mem_core = LSTM.lstm(self.mem_input_size , 1, self.mem_size, self.mem_layers, dropout) -- 1 here for output size
  self:_initMemory(1) -- initialize the memory 

  --Create the location and soft attention modules
  self.loc_att = nn.LocAttention(opt)
  self.soft_att = SOFT_ATT.softAttention(opt)

  -- create the core lstm network. note +1 for both the START and END tokens
  self.coreVid   = LSTM.lstm(self.input_encoding_size, 0, self.rnn_size, self.num_layers, dropout)
  self.coreLang  = LSTM_3D.lstm(self.embed_size, self.vocab_size + 1, self.rnn_size, self.num_layers, dropout, self.mem_size )

  self.lookup_table = nn.LookupTable(self.vocab_size + 1, self.embed_size)
  self:_createInitState(1) -- will be lazily resized later during forward passes
  
  -- variables for the soft attention  
  self.dumm_alpha = torch.Tensor()
  self.S_encoder = torch.Tensor()
  self.dS_encoder = torch.Tensor()
  self.dimgs  = torch.Tensor()

  -- variable for memory
  self.dummy_mem = torch.Tensor()

  --init hidden states and memery with a network 
  self.init_state_net_h0 = INIT_STATE_NN.initStateNN(opt)
  self.init_state_net_c0 = INIT_STATE_NN.initStateNN(opt)

end

function layer:_createInitState(batch_size)
  --[[
      Since it is LSTM, we need to create C(memeory) and Hidden states for each layer
      first create the C(t), then h(t)
    ]]
  assert(batch_size ~= nil, 'batch size must be provided')
  
  -- construct the initial state for the LSTM
  if self.init_state == nil then
      self.init_state = {}
  end
  if self.zero_state == nil then
      self.zero_state = {}
  end
  for h = 1, self.num_layers * 2 do  -- num_layers * 2 because we need one for C and one for H
    -- note, the init state Must be zeros because we are using init_state to init grads in backward call too
    if self.init_state[h] then

      if self.init_state[h]:size(1) ~= batch_size then
        self.init_state[h]:resize(batch_size, self.rnn_size):zero() --If we need to update to the current batch size 
        self.zero_state[h]:resize(batch_size, self.rnn_size):zero() --If we need to update to the current batch size 

      end
    else
      self.init_state[h] = torch.zeros(batch_size, self.rnn_size)
      self.zero_state[h] = torch.zeros(batch_size, self.rnn_size)

    end
  
  end
  
  self.num_state = #self.zero_state
end

function layer:createClones()
  --[[
      Construct the net clones 
      for each time t create a network 
   ]]
  print('Constructing clones inside the SeqToSeqLocSoftMeMAtt_sNN_R2')

  self.VidClones = {self.coreVid}
  self.LangClones = {}
  self.lookup_tables = {}
  self.loc_atts = {self.loc_att}
  self.soft_atts = {} -- we don't need to clone this for the video part

  for t = 2, self.total_seq_len  do  --includes start symbols, video sequence, 

    if ( t <= self.video_length) then
        self.VidClones[t] = self.coreVid:clone('weight', 'bias', 'gradWeight', 'gradBias')
        self.loc_atts[t] = self.loc_att:clone('weight', 'bias', 'gradWeight', 'gradBias')
    else 
        self.LangClones[t] = self.coreLang:clone('weight', 'bias', 'gradWeight', 'gradBias')
        self.lookup_tables[t] = self.lookup_table:clone('weight', 'gradWeight')
        self.soft_atts[t] = self.soft_att:clone('weight', 'bias', 'gradWeight', 'gradBias')
    end

  end
  -- clone the memeory as well
  self:createMemClones()

end

function layer:createMemClones()
    --[[
        Construct the net clones for Memory Network 
        for each time t create a network 
     ]]
    print('Constructing clones for Memory Network')
    self.mem_clones = {}

    for t = self.video_length + 2, self.total_seq_len do  --it creates a clone as long as input seq_length, it starts from index video_length + 1 (start word) + 1  
      self.mem_clones[t] = self.mem_core:clone('weight', 'bias', 'gradWeight', 'gradBias')
    end

end

function layer:_initMemory(batch_size)
  --[[
      Since it is LSTM, we need to create C(memeory) and Hidden states for each layer
      first creat the C(t), then h(t)
    ]]
  
  assert(batch_size ~= nil, 'batch size must be provided')
  
  -- construct the initial state for the LSTM
  if self.init_mem == nil then
      self.init_mem = {}
  end

  for h = 1, self.mem_layers * 2 do  -- num_layers * 2 because we need one for C and one for H
    -- note, the init state Must be zeros because we are using init_state to init grads in backward call too
    if self.init_mem[h] then

      if self.init_mem[h]:size(1) ~= batch_size then
        self.init_mem[h]:resize(batch_size, self.mem_size):zero() --If we need to update to the current batch size 
      end
    
    else
      self.init_mem[h] = torch.zeros(batch_size, self.mem_size)
    end
  
  end
  
  self.num_state_mem = #self.init_mem
end



function layer:getModulesList()
  return {self.coreVid, self.coreLang, self.lookup_table, self.loc_att, self.soft_att, self.mem_core, self.init_state_net_h0, self.init_state_net_c0}
end

function layer:parameters()
  --[[
      we only have two internal modules, 
      return their params
   ]]
  local p1, g1 = self.coreVid:parameters()
  local p2, g2 = self.coreLang:parameters()
  local p3, g3 = self.lookup_table:parameters()
  local p4, g4 = self.loc_att:parameters()
  local p5, g5 = self.soft_att:parameters()
  local p6, g6 = self.mem_core:parameters()
  local p7, g7 = self.init_state_net_h0:parameters()
  local p8, g8 = self.init_state_net_c0:parameters()

  -- Model Parameters 
  local params = {} 
  for k,v in pairs(p1) do   --coreVid
    table.insert(params, v) 
  end

  for k,v in pairs(p2) do   --coreLang
    table.insert(params, v) 
  end

  for k,v in pairs(p3) do    --Lookuptable
    table.insert(params, v) 
  end

  for k,v in pairs(p4) do   --Location attention
    table.insert(params, v) 
  end

  for k,v in pairs(p5) do   --Soft attention
    table.insert(params, v) 
  end

  for k,v in pairs(p6) do   --Memory
    table.insert(params, v) 
  end

  for k,v in pairs(p7) do  --init_SS_h0
    table.insert(params, v) 
  end

  for k,v in pairs(p8) do  --init_SS_C0
    table.insert(params, v) 
  end

  --Grad parameters
  local grad_params = {}
  for k,v in pairs(g1) do      
    table.insert(grad_params, v) 
  end

  for k,v in pairs(g2) do     
    table.insert(grad_params, v) 
  end

  for k,v in pairs(g3) do    
    table.insert(grad_params, v) 
  end

  for k,v in pairs(g4) do    
    table.insert(grad_params, v) 
  end

  for k,v in pairs(g5) do     
    table.insert(grad_params, v) 
  end

  for k,v in pairs(g6) do    
    table.insert(grad_params, v) 
  end

  for k,v in pairs(g7) do     
    table.insert(grad_params, v) 
  end

  for k,v in pairs(g8) do     
    table.insert(grad_params, v) 
  end
 
  return params, grad_params
end

function layer:training()
  --[[ 
      useful when using droupt set self.train = true 
      in nn.Module
    ]]
  if self.VidClones == nil or self.LangClones == nil then 
      self:createClones() 
  end 

  self.coreVid:training() 
  self.coreLang:training() 
  self.lookup_table:training() 
  self.loc_att:training() 
  self.soft_att:training() 
  self.mem_core:training() 

  for k,v in pairs(self.VidClones) do 
      v:training() 
  end

  for k,v in pairs(self.LangClones) do 
      v:training() 
  end

  for k,v in pairs(self.lookup_tables) do 
    v:training() 
  end

  for k,v in pairs(self.mem_clones) do 
    v:training() 
  end

  for k,v in pairs(self.loc_atts) do --not required, just in case
      v:training() 
  end

  for k,v in pairs(self.soft_atts) do -- no dropout but just in case
      v:training() 
  end

end

function layer:evaluate()
  -- useful when using droupt set self.train = false in nn.Module
  if self.VidClones == nil or self.LangClones == nil then 
      self:createClones() 
  end 
  self.coreVid:evaluate() 
  self.coreLang:evaluate() 
  self.lookup_table:evaluate() 
  self.loc_att:evaluate() 
  self.soft_att:evaluate() 
  self.mem_core:evaluate() 

  for k,v in pairs(self.VidClones) do 
      v:evaluate() 
  end

  for k,v in pairs(self.LangClones) do 
      v:evaluate() 
  end

  for k,v in pairs(self.lookup_tables) do 
    v:evaluate() 
  end

  for k,v in pairs(self.mem_clones) do 
    v:evaluate() 
  end

  for k,v in pairs(self.loc_atts) do -- no dropout but just in case
      v:evaluate() 
  end

  for k,v in pairs(self.soft_atts) do -- no dropout but just in case
      v:evaluate() 
  end

end

function layer:_createStateBuffers(batch_size)
  -- construct the S_encoder keep the states of encoder
  -- This will be used in softattention
  self.dummy_mem:resize(batch_size, self.mem_size):zero() -- will be used on encoder side, it never be updated
  self.dumm_alpha:resize(batch_size, self.video_length):zero() -- will be used on encoder side, it never be updated 
  self.S_encoder:resize(batch_size, self.video_length, self.rnn_size):zero()

end

--[[
input is a tuple of:
1. torch.Tensor of size NxK (K is dim of image code)
2. torch.LongTensor of size DxN, elements 1..M
   where M = opt.vocab_size and D = opt.seq_length

returns a (D+2)xNx(M+1) Tensor giving (normalized) log probabilities for the 
next token at every iteration of the LSTM (+2 because +1 for first dummy 
img forward, and another +1 because of START/END tokens shift)
--]]
function layer:updateOutput(input)
  local imgs = input[1] -- is video_len*batch_size*dim*conv_loc_size e.g. 128*16*512*196
  local seq  = input[2] -- seq is seq_length*batch_size

  if self.VidClones == nil or self.LangClones == nil then 
    self:createClones() -- lazily create clones on first forward pass
  end 

  assert(seq:size(1) == self.seq_length)
  local batch_size = seq:size(2)
  -- out of LSTM is h(t)[batch_size*rnn_size],c(t)[batch_size*rnn_size], log values(batch_size*vocab_size + 1) 
  -- so we need to keep the results for seq_length + 2 iteration. Remember, we ignore the fist output since 
  -- it is from image  
  self.output:resize(self.total_seq_len, batch_size, self.vocab_size + 1):zero() -- +1 since start and end are the same word
  self:_createStateBuffers(batch_size)
  self:_initMemory(batch_size)

  self.mem_state = {[self.video_length + 1] = self.init_mem}
  self.inputs = {}
  self.mem_inputs = {}
  self.lookup_tables_inputs = {}
  -------------------------------------------------------------
  -------------------------------------------------------------
  -- Now we need to initlize the h0 and c0 with a 2 layer feedforward network. No need to init every layer 
  self:_createInitState(batch_size)
  local h0_init_nn = self.init_state_net_h0:forward(imgs)
  local c0_init_nn = self.init_state_net_c0:forward(imgs)
  self.state = {[0] = self.init_state}

  -- only init the first layer with init_state_nn
  self.state[0][1] = c0_init_nn -- init c0 with the init_state_nn
  self.state[0][2] = h0_init_nn -- init h0 with the init_state_nn
  --self.init_state = {} -- free the memory, freeing memory is problematic when using cuda
  -------------------------------------------------------------
  -------------------------------------------------------------
  self.skip_dict = {}

  for t = 1, self.total_seq_len do

      local can_skip = false
      local xt, mem_output 
      mem_output = self.dummy_mem:clone() -- would be consider as S_hat that we had 

      if t <= self.video_length then
          -- feed in the images into the location attention
          local xt_img = imgs[t]
          -------------------------------------------------------------
          -------------------------------------------------------------
          -- Important: The LSTM states are C(t) and H(t). However, when we work with attention
          -- we only work with H(t)_L not C(t) where L == num_layer. The state variable frst 
          -- specify C(t) and then H(t)
          -- For example: num_layer = 2 init_state(1)==> c(t)_1, init_state(2)==> h(t)_1, 
          --                            init_state(3)==> c(t)_2, init_state(4)==> h(t)_2 [_L means layer]
          -------------------------------------------------------------
          -------------------------------------------------------------
          local img_hidden = {xt_img, self.state[t - 1][self.num_layers * 2]}  -- for t = 1, if would be all zeros
          xt = self.loc_atts[t]:forward(img_hidden) -- each image is batch_size*dim*conv_map (example: 128*512*196) ==> output: 128*512

          self.inputs[t] = {xt, unpack(self.state[t - 1])} 
          -- forward the network through lstm
          local out = self.VidClones[t]:forward(self.inputs[t])
          self.state[t] = {} -- current state ( contains all h(t-1_, c(t-1), update with c(t), h(t)

          for i = 1, self.num_state do 
            table.insert(self.state[t], out[i]) 
          end

          -- save the state of encoder 
          if t <= self.video_length then
              self.S_encoder[{{},{t},{}}] = self.state[t][self.num_layers * 2]:clone() 
          end

          table.insert(self.skip_dict, t)

      else

          if t == self.video_length + 1 then -- Done with videos, start the "start" token
              -- feed in the start token
              local it = torch.LongTensor(batch_size):fill(self.vocab_size + 1)
              self.lookup_tables_inputs[t] = it
              xt = self.lookup_tables[t]:forward(it) -- NxK sized input (token embedding vectors)

          else
              -- feed in the rest of the sequence...
              local it = seq[t - self.video_length - 1]:clone()
              if torch.sum(it) == 0 then
                  -- computational shortcut for efficiency. All sequences have already terminated and only
                  -- contain null tokens from here on. We can skip the rest of the forward pass and save time
                  -- Remeber the seq matrix is N*max seq length that for some sequence some of elements are zeros
                  can_skip = true 
              end
              --[[        
                seq may contain zeros as null tokens, make sure we take them out to any arbitrary token
                that won't make lookup_table crash with an error.
                token #1 will do, arbitrarily. This will be ignored anyway
                because we will carefully set the loss to zero at these places
                in the criterion, so computation based on this value will be noop for the optimization.
              --]]
              it[torch.eq(it, 0)] = 1

              if not can_skip then
                  self.lookup_tables_inputs[t] = it
                  xt = self.lookup_tables[t]:forward(it)
                  -------------------------------------------------------------------
                  -------------------------------------------------------------------
                  -- soft attention will be calucalte for all (S, H(t-1)) and create 
                  -- an x_t hat which should be added to the decoder LSTM 
                  -------------------------------------------------------------------
                  -------------------------------------------------------------------
                  local t_mem = t 
                  local H = self.state[t - 1][self.num_layers * 2]:clone() 
                  local h_M = self.mem_state[t_mem - 1][self.num_state_mem]:clone() -- we only need not H in Soft Attention
                  -- input H, S, Memory state, output is s_hat (output[2]) which is B * K and alpha(output(1))  
                  local att_ouput = self.soft_atts[t]:forward({H, self.S_encoder, h_M}) 
                  self.mem_inputs[t_mem] = {att_ouput[2], unpack(self.mem_state[t_mem - 1]) } 

                  -------------------------------------------------------------------
                  -- now update the memory
                  local m_outputs = self.mem_clones[t_mem]:forward(self.mem_inputs[t_mem]) -- input B*K --> output B*M
                  self.mem_state[t_mem] = {} -- current state ( contains all h(t-1_, c(t-1), update with c(t), h(t)
                  for j = 1, self.num_state_mem do 
                     table.insert(self.mem_state[t_mem], m_outputs[j]) 
                  end
                  mem_output  =  m_outputs[self.num_state_mem] -- first output is C(t), so we need 2nd output to get H(t)
              end
          end
          if not can_skip then
              
              -- construct the inputs
              self.inputs[t] = {mem_output, xt, unpack(self.state[t - 1])}  -- for t = 1, state would be all zeros
              -- forward the network through lstm
              local out = self.LangClones[t]:forward(self.inputs[t])
              -- process the outputs
              self.output[t] = out[self.num_state + 1] -- last element is the output vector, others are C(t-1) and H(t-1) for each layer
              self.state[t] = {} -- current state ( contains all h(t-1_, c(t-1), update with c(t), h(t)
              
              for i = 1, self.num_state do 
                table.insert(self.state[t], out[i]) 
              end

              table.insert(self.skip_dict, t)
              
          end
      end

  end
  
  return self.output

end


function layer:updateGradInput(input, gradOutput)
  -- input same size as input for forward
  -- gradOutput is ([seq_length + 1][batch][Voca +_1] )
  --remember self.init_state[0] is all zeros

  local input_img= input[1]  

  -- input_image is video_len*bacth_size*input_encoding_size
  self.dimgs:resize(input_img:size(1), input_img:size(2), input_img:size(3), input_img:size(4)):zero() -- grad on input images
  self.dS_encoder:resizeAs(self.S_encoder):zero()

  local dstate = {[self.skip_dict[#self.skip_dict]] = self.zero_state} -- this works when init_state is all zeros
  local dstate_mem = {[self.skip_dict[#self.skip_dict]] = self.init_mem} -- this works when init_state is all zeros

  -- go backwards and compute gradients
  for t0 = #self.skip_dict, 1, -1 do
      t = self.skip_dict[t0]

      -- concat state gradients and output vector gradients at time step t
      local dout = {}
      
      for k = 1, #dstate[t] do 
        table.insert(dout, dstate[t][k]) 
      end

      dstate[t - 1] = {} -- copy over rest to state grad
      if ( t > self.video_length) then

          local dinputs, dxt
          table.insert(dout, gradOutput[t]) -- the output for video part is zero
          dinputs = self.LangClones[t]:backward(self.inputs[t], dout) -- self.inputs[t] contains x_t and h(t-1) and c(t-1)
          dxt = dinputs[2] -- first element is s_hat vector, and second element is xt

          for k = 2, self.num_state + 1 do  -- + 1 for grad dout, look above for loop and table.insert(dout, gradOutput[t])
              table.insert(dstate[t - 1], dinputs[k + 1] )  --> + 1 becuase inputs is {x_hat, xt, unpack(self.state[t - 1]) } 
          end

          if t ~= self.video_length + 1  then -- no attention of start token
              --------------------------------------------------------
              -- Update Memory 
              --------------------------------------------------------
              local t_mem = t 
              local dout_mem = {}

              for k = 1, self.num_state_mem do 
                  table.insert(dout_mem, dstate_mem[t_mem][k]:clone()) 
              end
    
              dout_mem[self.num_state_mem]:add(dinputs[1])
              local dmem_input = self.mem_clones[t_mem]:backward(self.mem_inputs[t_mem], dout_mem) --self.inputs[t] contains s_hat, x_t, c(t-1), and h(t-1)

              dstate_mem[t_mem - 1] = {}
              for k = 1, self.num_state_mem do
                  table.insert(dstate_mem[t_mem - 1], dmem_input[k + 1]:clone())  --> + 1 becuase inputs is {att_ouput[2], unpack(self.mem_state[t_mem - 1]) } 
              end

              --------------------------------------------------------
              --The gradient with respect to the H(t) and S_encoder
              --------------------------------------------------------
              local H = self.inputs[t][self.num_layers * 2 + 2] -- inputs is {x_hat, xt, unpack(self.state[t - 1]) } -- refer to LSTM_3D 
              local h_M = self.mem_inputs[t_mem][ self.num_state_mem + 1 ] -- input is {att_ouput[2], unpack(self.mem_state[t_mem - 1]) } 
              local dsoft_attn = self.soft_atts[t]:backward({H, self.S_encoder, h_M}, {self.dumm_alpha, dmem_input[1]})
              dstate[t - 1][self.num_layers * 2]:add(dsoft_attn[1])
              self.dS_encoder:add(dsoft_attn[2])
              dstate_mem[t_mem - 1][self.num_state_mem]:add(dsoft_attn[3])
          end    

          local it = self.lookup_tables_inputs[t]
          local dummy = self.lookup_tables[t]:backward(it, dxt) -- backprop into lookup table

      else
          local dinputs, dxt
          dout[self.num_layers * 2]:add(self.dS_encoder[{{},{t},{}}]) 
          dinputs = self.VidClones[t]:backward(self.inputs[t], dout) -- self.inputs[t] contains x_t and h(t-1) and c(t-1)
          
          for k = 2, self.num_state + 1 do  -- + 1 for grad dout, look above for loop and table.insert(dout, gradOutput[t])
              table.insert(dstate[t - 1], dinputs[k]) 
          
          end
          
          dxt = dinputs[1] -- first element is the input vector
          local input_attn = {input_img[t], self.inputs[t][self.num_layers * 2]} --input_img[t] is input images and self.inputs[t][2] are states 
          local dloc_att = self.loc_atts[t]:backward(input_attn, dxt) 
          self.dimgs[{{t,t},{},{}}] = dloc_att[1]
          -------------------------------------------------
          --  The gradient with respect to the dH_L should be updated as well by 
          --  gradient backprop from attention. I've explained in the updateOutput method
          --  in the first if's body  
          ----
          dstate[t - 1][self.num_layers * 2]:add(dloc_att[2])  

      end
  end
  --update the initlization
  self.dimgs:add(self.init_state_net_c0:backward(self.inputs[1], dstate[0][1]))
  self.dimgs:add(self.init_state_net_h0:backward(self.inputs[1], dstate[0][2]))

  -- we have gradient on image, but for LongTensor gt sequence we only create an empty tensor - can't backprop
  self.gradInput = {self.dimgs, torch.Tensor()}
  return self.gradInput
end

function layer:clearState()
  self.dimgs:set()
  self.dumm_alpha:set() -- will be used on encoder side, it never be updated 
  self.S_encoder:set()
  self.dS_encoder:set()
  self.dummy_mem:set()
end
----------------------------------------------------------------------
----------------------Sampling and beam_search------------------------
----------------------------------------------------------------------
--[[
takes a batch of images and runs the model forward in sampling mode
Careful: make sure model is in :evaluate() mode if you're calling this.
Returns: a DxN LongTensor with integer elements 1..M, 
where D is sequence length and N is batch (so columns are sequences)
--]]
function layer:sample(imgs, opt)
  local sample_max = utils.get_kwarg(opt, 'sample_max', 1)
  local beam_size = utils.get_kwarg(opt, 'beam_size', 1)
  local temperature = utils.get_kwarg(opt, 'temperature', 1.0)
  
  if sample_max == 1 and beam_size > 1 then 
    return self:sample_beam(imgs, opt) 
  end -- indirection for beam search
   
  local batch_size = imgs:size(2) -- image is video_length*batch_size*encoding_size
  self:_createInitState(batch_size)
  self:_createStateBuffers(batch_size)
  self:_initMemory(batch_size)
  local mem_state =  self.init_mem

  local state = self.init_state
  local h0_init_nn = self.init_state_net_h0:forward(imgs)
  local c0_init_nn = self.init_state_net_c0:forward(imgs)

  -- only init the first layer with init_state_nn
  state[1] = c0_init_nn -- init c0 with the init_state_nn
  state[2] = h0_init_nn -- init h0 with the init_state_nn

  -- we will write output predictions into tensor seq
  local seq = torch.LongTensor(self.seq_length, batch_size):zero()
  local seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
  local logprobs -- logprobs predicted in last time step

  for t = 1, self.total_seq_len do

      local xt, it, sampleLogprobs, mem_output 
      mem_output = self.dummy_mem:clone() -- would be consider as S_hat that we had 

      if t <= self.video_length then
          -- feed in the images, here we only feed image at time, 
          -- if we want to feed image to each time steps , we just need adjust the t <= #frames  
          local xt_img = imgs[t]
          local img_hidden = {xt_img, state[self.num_layers * 2]}  -- for t = 1, if would be all zeros
          xt = self.loc_att:forward(img_hidden) -- each image is batch_size*dim*conv_map (example: 128*512*196) ==> output: 128*512
          local inputs = {xt, unpack(state)} 
          local out = self.coreVid:forward(inputs)

          state = {}

          for i = 1, self.num_state do 
              table.insert(state, out[i]) 
          end
          self.S_encoder[{{},{t},{}}] = state[self.num_layers * 2]:clone() 

      else
          if t == self.video_length + 1 then
              -- feed in the start tokens
              it = torch.LongTensor(batch_size):fill(self.vocab_size + 1)
              xt = self.lookup_table:forward(it)
          else
              -- First needs to figure out what would be the next "it"
              -- to feed into look_up table
              -------------------------------------
              -- take predictions from previous time step and feed them in
              if sample_max == 1 then
                -- use argmax "sampling"
                sampleLogprobs, it = torch.max(logprobs, 2)
                it = it:view(-1):long()
              else
                -- sample from the distribution of previous predictions
                local prob_prev
                if temperature == 1.0 then
                  prob_prev = torch.exp(logprobs) -- fetch prev distribution: shape Nx(M+1)
                else
                  -- scale logprobs by temperature
                  prob_prev = torch.exp(torch.div(logprobs, temperature))
                end
                it = torch.multinomial(prob_prev, 1)
                sampleLogprobs = logprobs:gather(2, it) -- gather the logprobs at sampled positions
                it = it:view(-1):long() -- and flatten indices for downstream processing
              end
              -------------------------------------
              xt = self.lookup_table:forward(it)

              -------------------------------------------------------------------
              ----Apply soft attention
              -------------------------------------------------------------------
              local t_mem = t 
              local H = state[self.num_layers * 2]:clone()
              local h_M = mem_state[self.num_state_mem] -- we only need not H in Soft Attention
              local att_ouput = self.soft_att:forward({H, self.S_encoder, h_M}) -- input H, S and output is s_hat (output[2]) which is B * K and alpha(output(1))
              local mem_inputs = {att_ouput[2], unpack(mem_state) } 
              
              -------------------------------------------------------------------
              -- now update the memory
              local m_outputs = self.mem_core:forward(mem_inputs) -- input B*K --> output B*M
              mem_state = {} -- current state ( contains all h(t-1_, c(t-1), update with c(t), h(t)
              for j = 1, self.num_state_mem do 
                   table.insert(mem_state, m_outputs[j]) 
              end
              mem_output  =  m_outputs[self.num_state_mem] -- first output is C(t), so we need 2nd output to get H(t)
          end

          local inputs = {mem_output, xt, unpack(state)}  -- Remember the inputs are X(t) images and the C, h into LSTM 
          local out = self.coreLang:forward(inputs)
          logprobs = out[self.num_state + 1] -- last element is the output vector
          
          state = {}
          for i = 1, self.num_state do 
              table.insert(state, out[i]) 
          end

      end

      if t >= self.video_length + 2 then 
        seq[t - self.video_length - 1] = it -- record the samples
        seqLogprobs[t - self.video_length - 1] = sampleLogprobs:view(-1):float() -- and also their log likelihoods
      end

  end

  -- return the samples and their log likelihoods
  return seq, seqLogprobs
end

--[[
Modiefed version of https://github.com/karpathy/neuraltalk2/blob/master/misc/LanguageModel.lua
]]--
function layer:sample_beam(imgs, opt)
  local beam_size = utils.get_kwarg(opt, 'beam_size', 10)
  --image is video_length*batch_size*encoding_size
  local batch_size, feat_dim = imgs:size(2), imgs:size(3)
  local function compare(a,b) return a.p > b.p end -- used downstream

  assert(beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed')

  local seq = torch.LongTensor(self.seq_length, batch_size):zero()
  local seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)

  -- lets process every image independently for now, for simplicity
  for k = 1, batch_size do

      -- create initial states for all beams
      self:_createInitState(beam_size)
      self:_createStateBuffers(beam_size)
      self:_initMemory(beam_size)

      local state = self.init_state
      local mem_state =  self.init_mem

      -- we will write output predictions into tensor seq
      local beam_seq = torch.LongTensor(self.seq_length, beam_size):zero()
      local beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size):zero()
      local beam_logprobs_sum = torch.zeros(beam_size) -- running sum of logprobs for each beam
      local logprobs -- logprobs predicted in last time step, shape (beam_size, vocab_size+1)
      local done_beams = {}
          
      for t = 1, self.total_seq_len do

          local xt, it, sampleLogprobs, mem_output
          local new_state
          mem_output = self.dummy_mem:clone()

          if t == 1 then
              local h0_init_nn
              local c0_init_nn 
              local img_0 = imgs[{{},{k,k},{},{}}]
              local xt_img = img_0:repeatTensor(1 ,beam_size, 1, 1)-- k'th image feature expanded 
              h0_init_nn  = self.init_state_net_h0:forward(xt_img)
              c0_init_nn  = self.init_state_net_c0:forward(xt_img)
              -- only init the first layer with init_state_nn
              state[1] = c0_init_nn:clone() -- init c0 with the init_state_nn
              state[2] = h0_init_nn:clone() -- init h0 with the init_state_nn
          end  

          if t <= self.video_length then

                -- feed in the images, here we only feed one video at time, 
                -- if we want to feed image to each time steps , we just need adjust the t <= #frames 
                local img_0 = imgs[t] 
                local img_1 = img_0[k]  
                local xt_img = img_1:repeatTensor(beam_size,1,1)-- k'th image feature expanded ou
                local img_hidden = {xt_img, state[self.num_layers * 2]}  -- for t = 1, if would be all zeros
                xt = self.loc_att:forward(img_hidden) -- each image is batch_size*dim*conv_map (example: 128*512*196) ==> output: 128*512
          else
                if t == self.video_length + 1 then
                    -- feed in the start tokens
                    it = torch.LongTensor(beam_size):fill(self.vocab_size + 1)
                    xt = self.lookup_table:forward(it)
                else
                    --[[
                      perform a beam merge. that is,
                      for every previous beam we now many new possibilities to branch out
                      we need to resort our beams to maintain the loop invariant of keeping
                      the top beam_size most likely sequences.
                    ]]--
                    local logprobsf = logprobs:float() -- lets go to CPU for more efficiency in indexing operations
                    ys,ix = torch.sort(logprobsf, 2, true) -- sorted array of logprobs along each previous beam (last true = descending)
                    local candidates = {}
                    local cols = math.min(beam_size,ys:size(2))
                    local rows = beam_size
                    if t == self.video_length + 2 then -- at first time step only the first beam is active
                      rows = 1 
                    end 
                    for c = 1, cols do -- for each column (word, essentially)
                      for q = 1, rows do -- for each beam expansion
                        -- compute logprob of expanding beam q with word in (sorted) position c
                        local local_logprob = ys[{ q, c }]
                        local candidate_logprob = beam_logprobs_sum[q] + local_logprob
                        table.insert(candidates, {c=ix[{ q, c }], q=q, p=candidate_logprob, r=local_logprob })
                      end
                    end
                    table.sort(candidates, compare) -- find the best c,q pairs 

                    -- construct new beams
                    new_state = net_utils.clone_list(state)
                    local beam_seq_prev, beam_seq_logprobs_prev
                    if t > self.video_length + 2  then
                      -- well need these as reference when we fork beams around
                      beam_seq_prev = beam_seq[{ {1, t - (self.video_length + 2)}, {} }]:clone()
                      beam_seq_logprobs_prev = beam_seq_logprobs[{ {1, t - (self.video_length + 2)}, {} }]:clone()
                    end
                    for vix = 1, beam_size do
                        local v = candidates[vix]
                        -- fork beam index q into index vix
                        if t > self.video_length + 2  then
                          beam_seq[{ {1,t - (self.video_length + 2)}, vix }] = beam_seq_prev[{ {}, v.q }]
                          beam_seq_logprobs[{ {1, t - (self.video_length + 2)}, vix }] = beam_seq_logprobs_prev[{ {}, v.q }]
                        end
                        -- rearrange recurrent states
                        for state_ix = 1, #new_state do
                          -- copy over state in previous beam q to new beam at vix
                          new_state[state_ix][vix] = state[state_ix][v.q]
                        end
                        -- append new end terminal at the end of this beam
                        beam_seq[{ t - (self.video_length + 1), vix }] = v.c -- c'th word is the continuation
                        beam_seq_logprobs[{ t - (self.video_length + 1) , vix }] = v.r -- the raw logprob here
                        beam_logprobs_sum[vix] = v.p -- the new (sum) logprob along this beam

                        if v.c == self.vocab_size + 1 or t == self.total_seq_len then
                          -- END token special case here, or we reached the end.
                          -- add the beam to a set of done beams
                          table.insert(done_beams, {seq = beam_seq[{ {}, vix }]:clone(), 
                                                    logps = beam_seq_logprobs[{ {}, vix }]:clone(),
                                                    p = beam_logprobs_sum[vix]
                                                   })
                        end
                    end
                    -- encode as vectors
                    it = beam_seq[ t -  (self.video_length + 1)]
                    xt = self.lookup_table:forward(it)    
                end    
          end 
          
          if new_state then 
             state = new_state 
          end -- swap rnn state, if we reassinged beams

          local out 
          if t <= self.video_length then
              local inputs = {xt, unpack(state)} 
              out = self.coreVid:forward(inputs)
              state = {}
              for i=1, self.num_state do 
                  table.insert(state, out[i]) 
              end
              self.S_encoder[{{},{t},{}}] = state[self.num_layers * 2]:clone() 

          else 
              if t > self.video_length + 1 then 

                  local t_mem = t 
                  local H = state[self.num_layers * 2]:clone()
                  local h_M = mem_state[self.num_state_mem] -- we only need not H in Soft Attention
                  local att_ouput = self.soft_att:forward({H, self.S_encoder, h_M}) -- input H, S and output is s_hat (output[2]) which is B * K and alpha(output(1))
                  local mem_inputs = {att_ouput[2], unpack(mem_state) } 
                  
                  -------------------------------------------------------------------
                  -- now update the memory
                  local m_outputs = self.mem_core:forward(mem_inputs) -- input B*K --> output B*M
                  mem_state = {} -- current state ( contains all h(t-1_, c(t-1), update with c(t), h(t)
                  for j = 1, self.num_state_mem do 
                       table.insert(mem_state, m_outputs[j]) 
                  end
                  mem_output  =  m_outputs[self.num_state_mem] -- first output is C(t), so we need 2nd output to get H(t)
            end  

                local inputs = {mem_output, xt, unpack(state)}
                out = self.coreLang:forward(inputs)
                logprobs = out[self.num_state+1] -- last element is the output vector
                state = {}
                for i=1, self.num_state do 
                    table.insert(state, out[i]) 
                end

          end  
      end   

      table.sort(done_beams, compare)
      seq[{ {}, k }] = done_beams[1].seq -- the first beam has highest cumulative score
      seqLogprobs[{ {}, k }] = done_beams[1].logps
  end

  -- return the samples and their log likelihoods
  return seq, seqLogprobs
end
