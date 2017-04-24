require 'nn'
require 'nngraph'
local utils = require 'misc.utils_helper'

-------------------------------------------------------------------------------
-- Some notes
-------------------------------------------------------------------------------
--[[
     Create the attention module
     input S \in B*L*K and H \in B*K where S is the hidden states for the encoder and 
     H is decoder hidden states. S_hat is calcuated at the previous time step and it comes from Mmeory/LSTM. 
     M = tanh( S * W_s   + H_expand(L)* W_h + b  + S_hat * Wz) where W_s is K*K, W_h is K*K,
     ,W_z is M*K and M \in B*L*K.
     alpha = softmax(M*u) where u is K*1 and alpha would be B * L
     c =  alpha:t()*S where c would be B*K 

     -- This is same as SoftAttention, the only difference is it gets three inputs rather than 2.
  ]]

local SoftAttention_3D = {}
function SoftAttention_3D.softAttention(opt)

    K =  utils.get_kwarg(opt, 'rnn_size')
    L =  utils.get_kwarg(opt, 'video_length')
    M =  utils.get_kwarg(opt, 'mem_size')

    local inputs = {}
    table.insert(inputs, nn.Identity()()) --B * K : this contains hidden units for decoder     -e.g. H
    table.insert(inputs, nn.Identity()()) --B * L *K  : this contains hidden units for encoder -e.g. S
    table.insert(inputs, nn.Identity()()) --B * M  : this contains hidden units from the memory -e.g. S_hat(t-1)

    local H = inputs[1]  -- B * K
    local S = inputs[2]  -- B * L * K
    local Z = inputs[3]  -- B * M, we call S_hat(t-1) as Z


    local HW  = nn.Linear(K, K)(H):annotate{name = 'H*Wh'} -- (B *K) x (K * K) --> B*K 
    local HpW = nn.Replicate(L, 2)(HW)  -- replicate HW to B*L*K

    local ZW  = nn.Linear(M, K)(Z):annotate{name = 'Z*Wz'} -- (B *M) x (M * K) --> B*K 
    local ZpW = nn.Replicate(L, 2)(ZW)  -- replicate ZW to B*L*K

    local Sr   -- reshape S from B * L * K to BL * K
    Sr  = nn.View(-1, K)(S)-- reshape S from B * L * K to BL * K

    local SrW = nn.Linear(K, K)(Sr):annotate{name = 'S*Ws'} -- (BL *K) * (K * K) --> BL * K
    local SW  = nn.View(-1 , L, K)(SrW) -- reshape WSr from BL * K  to B * L * K

    -- now we have W*Hp and W*S lets sum them up then and apply tanh  => tanh( S*Ws + H*Wh)
    local dot = nn.CAddTable()({SW, HpW, ZpW})  --> output is B * L *K
    local tanh = nn.Tanh()(dot)
    
    -- now times to apply softmax
    local tanhr -- reshape tanhr from B * L *K  to BL * K
    tanhr  = nn.View(-1, K)(tanh) -- reshape tanhr from B * L *K  to BL * K

    local projr   = nn.Linear(K, 1)(tanhr):annotate{name = 'tanh*V'} -- (BL *K) * (K * 1) --> BL * 1
    local proj    = nn.View(-1 , L)(projr) -- BL * 1 --> B * L
    local alpha   = nn.SoftMax()(proj) --> B * L

    local S_hat0   = nn.MM(true, false)({ S, nn.Replicate(1, 3)(alpha)}) -- S(B * L * K ) x alpha(B * 1 * L) --> B * K * 1
    local S_hat1 =   nn.Squeeze()(S_hat0) --> B * K * 1 to B * K
    local outputs = {}
    table.insert(outputs, alpha) --> alpha is B * L
    table.insert(outputs, S_hat1) --> S_hat is B * K
 
    return nn.gModule(inputs, outputs)
end

return SoftAttention_3D
