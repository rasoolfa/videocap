require 'torch'
require 'nn'
local utils = require 'misc.utils_helper'
-------------------------------------------------------------------------------
-- Some notes
-------------------------------------------------------------------------------
--[[
     Create the attention layer over the location
     inputs:
          F \in R^{B*D*P} where P number of locations and D is input dimension and B is batch size
          S \in R^{B*K} where K is hidden unit dimension
      outputs:
          hat{X} \in R^{B * D} 
      formulation:
          X = \sum_i(l_i*F_i) where l_i is softmax(V_i^T*S). 
          V here is a learable parameters and is R \in {K*P} 
  ]]

local layer, parent = torch.class('nn.LocAttention', 'nn.Module')

function layer:__init(opt)
  parent.__init(self)
  self.K =  utils.get_kwarg(opt, 'rnn_size', 512)
  self.P =  utils.get_kwarg(opt, 'conv_loc_size', 196)
  --self.B =  utils.get_kwarg(opt, 'batch_size')
  self.D =  utils.get_kwarg(opt, 'input_encoding_size', 512) -- input dims
  
  -- input to this module is B*K 
  self.layer_softmax = nn.Sequential() 
  self.layer_softmax:add(nn.Linear(self.K , self.P))
  self.layer_softmax:add(nn.SoftMax()) --output B*P
  self.layer_softmax:add(nn.View(-1, self.P, 1)) --- convert B*P to B*P*1

  --add a model for calculation of [B*D*P]x[B*P*1] and gets [B*D*1]     
  self.prod = nn.MM() 
  self.view_prod = nn.View(-1, self.D)

end

function layer:updateOutput(input)
  -- init the parameters
  local F = input[1] --input, this one should be B * D * P
  local S = input[2] --hidden unit, this one should be B * K
  local B = F:size(1) --batch size

  assert(S:size(2) == self.K, 'Something is wrong with "Hidden" dimension')
  assert(F:size(3) == self.P, 'Something is wrong with "input" dimension')
 
  self.output:resize(B, self.D):zero()
  -----------------------------------
  -- Calculate the following:
  -- l_i = Softmax(v_i^t*S)
  -- S_hat = sum_i^p(l_i*F_i) 
  ------------------------------------
  self.alpha = self.layer_softmax:forward(S) --> input(B*K)--> output B*P*1
  self.out_prod = self.prod:forward({F, self.alpha}) --(B * D * P) * (B * P * 1)  ==> output is (B * D * 1)
  self.output = self.view_prod:forward(self.out_prod) --> input (B * D * 1) ==> output (B * D )

  return self.output

end

function layer:updateGradInput(input, gradOutput)
  -- gradOutput is B * D and inputs are as follows
  local F = input[1] --input, this one should be B * D * P
  local S = input[2] --hidden unit, this one should be B * K 
  ----
  --  gradient calculation
  ----
  local dview = self.view_prod:backward( self.out_prod, gradOutput) --output (B * D * 1) 
  local grad_prod = self.prod:backward( {F, self.alpha}, dview)
  local dF, dalpha = grad_prod[1], grad_prod[2]     -- ==> B * D * P, B * P * 1   
  local dS = self.layer_softmax:backward(S, dalpha) -- ==> B*K

  self.gradInput = {dF, dS}

  return self.gradInput

end

function layer:parameters()
  --[[
      set the parameters 
   ]]
  local params = {}
  local grad_params = {}
  local p1, g1 = self.layer_softmax:parameters()

  for k,v in pairs(p1) do 
      table.insert(params, v) 
  end
  for k,v in pairs(g1) do 
      table.insert(grad_params, v) 
  end

  return params, grad_params
end

function layer:getAlphas()
  return self.alpha
end

function layer:share(other,...)
  self.layer_softmax:share(other.layer_softmax,...) 
  self.prod:share(other.prod, ...) --no paramters, but lets do it 
  self.view_prod:share(other.view_prod,...) --no paramters, but lets do it 

end 


 
