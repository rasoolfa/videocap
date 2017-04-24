----------------------------------------------------------------------
-- This code is from 
-- https://github.com/karpathy/neuraltalk2/blob/master/misc/LSTM.lua 
----------------------------------------------------------------------
require 'nn'
require 'nngraph'

local LSTM = {}

function LSTM.lstm(input_size, output_size, rnn_size, num_layers, dropout)
  --[[
  Parameters:
    input_size = input_encoding_size
    output_size = vocab_size + 1 ==> note +1 for both the START and END tokens
    rnn_size = #num hidden units 
    num_layers = num_layers
    dropout  
    This LSTM works only for given time, not like the other implemntation 
         i = sigma (X_t * W_xi + h_t-1*W_hi )
         f = sigma (X_t * W_xf + h_t-1*W_hf )
         o = sigma (X_t *W_xo + h_t-1*W_ho )
         g = tanh (X_t *W_xg + h_t-1*W_hg  )
         c_t = ft . c_t-1 + i_t . g_t
         h_t = o_t . tanh(c_t)
         p_t+1 =  softmax(h_t)
    W_x will be [input_size, 4*rnn_size]
    W_h will be [rnn_size,   4*rnn_size]
    b will be [4*rnn_size]

    the inputs to this module have the following sizes (the inputs should be in following order as well): 
    X which is batch_size * input_size
    C(t-1) which is batch_size * rnn_size 
    H(t-1) which is batch_size * rnn_size    
    and the outputs are: 
    C(t) which is batch_size * rnn_size 
    H(t) which is batch_size * rnn_size 

  ]]
  dropout =  dropout or 0
  
  -- there will be 2*num_layers+1 inputs i.e. X,C,h ...
  local inputs = {}
  table.insert(inputs, nn.Identity()())  -- indices giving the sequence of symbols
  for L = 1, num_layers do
    table.insert(inputs, nn.Identity()())  -- prev_c[L]
    table.insert(inputs, nn.Identity()())  -- prev_h[L]
  end
    --- inputs[1] --> input
    --- inputs[2] --> prev_c
    --- inputs[3] --> prev_h

  local x, input_size_L 
  local outputs = {}

  for L = 1, num_layers do
    --c,h from previous time steps 
    local prev_h = inputs[L * 2 + 1] -- look at lines 31-38
    local prev_c = inputs[L * 2]     -- look at lines 31-38

    -- the input to this layer 
    if L == 1 then
      x = inputs[1]
      input_size_L = input_size
    else
      x = outputs[(L - 1 ) *2] 
      if dropout > 0 then
        x = nn.Dropout(dropout)(x):annotate{name='drop_' .. L }   --- This dropout applies to non-recurrent weights
      end
      input_size_L = rnn_size
    end
    --[[
         i = sigma (X_t*W_xi + h_t-1*W_hi +b_i)
         f = sigma (X_t*W_xf + h_t-1*W_hf +b_f)
         o = sigma (X_t*W_xo + h_t-1*W_ho +b_o)
         g = tanh (X_t*W_xg  + h_t-1*W_hg +b_g)
         concat all W_x and W_h across different gates
         c_t = ft . c_t-1 + i_t . g_t
         h_t = o_t . tanh(c_t)
      ]]

    ---W_xj*X_t + w_hj*h_t-1   
    local i2h  = nn.Linear(input_size_L, 4 * rnn_size)(x):annotate{name = 'i2h_'..L}
    local h2h  = nn.Linear(rnn_size,     4 * rnn_size)(prev_h):annotate{name = 'h2h_'..L}
    local all_input_sums = nn.CAddTable()({ i2h , h2h})

    local reshaped = nn.Reshape(4, rnn_size)(all_input_sums) -- all_input_sums is 4rnn_size*1 --> reshape 4*rnn_size
    --[[
      SplitTable(dimension, nInputDims)
      Creates a module that takes a Tensor as input 
      and outputs several tables, splitting the Tensor along the specified dimension
      The optional parameter nInputDims allows to specify the number of
      dimensions that this module will receive. This makes it possible to forward 
      both minibatch and non-minibatch Tensors through the same module
    ]]
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4) -- this split(4) makes them to be copied into n1,n2,...

    --decode gates
    local in_gate = nn.Sigmoid()(n1)
    local forget_gate = nn.Sigmoid()(n2)
    local out_gate =  nn.Sigmoid()(n3)
    --decode the write inputs 
    local in_transform =  nn.Tanh()(n4)

    -- perform LSTM update 
    --c_t = ft . c_t-1 + i_t . g_t
    local next_c  = nn.CAddTable()({ nn.CMulTable()({in_gate, in_transform}) , nn.CMulTable()({forget_gate, prev_c})  })

    --  h_t = o_t . tanh(c_t)
    local next_h  = nn.CMulTable()({ out_gate , nn.Tanh()(next_c) })

    table.insert(outputs, next_c)
    table.insert(outputs,next_h)
  end

  if output_size > 1 then  --sometimes we don't need to calucalte the output, just use LSTM as memeory 
      -- set up the decoder
      local top_h = outputs[#outputs] -- if L=1, outputs size 2, outputs[1] --> next_c, outputs[2] --> next_h
      if dropout > 0 then
        top_h = nn.Dropout(dropout)(top_h):annotate{name='drop_final'}
      end
      -- we do this lienar decoding to learn the prob of each vocabs again
      local proj = nn.Linear(rnn_size, output_size)(top_h):annotate{name='decoder'}
      local logsoft = nn.LogSoftMax()(proj)
      --logsoft is batch_size*output_size where output_size is vocab_size
      table.insert(outputs, logsoft)
  end
  
  return nn.gModule(inputs, outputs)
end

return LSTM 
