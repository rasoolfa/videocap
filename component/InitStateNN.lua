require 'nn'
require 'nngraph'
local utils = require 'misc.utils_helper'
-------------------------------------------------------------------------------
-- Some notes
-------------------------------------------------------------------------------
--[[
     This layer creates a two layer feedforward newtork that will be used to init the 
     first hidden(h0) and memory unit(h0).

     inputs:
          X0: Batch * frames * input_dim * location
      outputs:
          Output:  Batch * RNN_size
      formulation:
          X0 \in R^{L x B x M X P}
          X1 = mean(X0, 4) -- mean over 4th dimention
          X2 = mean(X1, 1) -- mean over 1st dimention
          Now X2 is B x M
          F1 = ReLU(X2 * W) where W is M*K where K <<<<< D
          F2 = ReLU(F1 * U) where U is K*D where K <<<<< D
          Here: W  and U are learnable parameters[No bias] 
  ]]
 local InitStateNN = {}
 function InitStateNN.initStateNN(opt)
    M  = utils.get_kwarg(opt, 'input_encoding_size')
    D =  utils.get_kwarg(opt, 'rnn_size')
    K =  utils.get_kwarg(opt, 'hidden_size_init', 200)
    
    local inputs = {}
    table.insert(inputs, nn.Identity()())  

    local X0 =  inputs[1]--- input is L x B x M X P
    local X1 = nn.Squeeze()(nn.Mean(4)(X0))      --- L x B x M X P ==> L x B x M x1 ==> L x B x M    
    local X2 = nn.Squeeze()(nn.Mean(1)(X1))      --- L x B x M  ==> 1 x B x M  ==> B x M
    local f1 = nn.Linear(M, K, false)(X2):annotate{name = 'X2*W'} -- [B x M] * [M * k] ==> B x K
    local tanh = nn.Tanh()(f1)
    local f2   = nn.Linear(K, D, false)(tanh):annotate{name = 'f1*U'} -- [B x K] * [K * D] ==> B x D
    local init_val = nn.Tanh()(f2)
    local outputs = {}
    table.insert(outputs, init_val)      
    
    return nn.gModule(inputs, outputs)
end

return InitStateNN