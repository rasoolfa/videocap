function adam(x, dfdx, config, state)
    --[[ 
        This is adam implementation based on the torch code,
        just changed a bit to reflect 
        our needs here.
        Source: https://github.com/torch/optim/blob/master/adam.lua
    ]]
    -- (0) get/update state
    local lr = config.learningRate or 0.001

    local beta1 = config.beta1 or 0.9
    local beta2 = config.beta2 or 0.999
    local epsilon = config.epsilon or 1e-8
    local wd = config.weightDecay or 0

    -- (2) weight decay
    if wd ~= 0 then
       dfdx:add(wd, x)
    end

    -- Initialization
    state.t = state.t or 0
    -- Exponential moving average of gradient values
    state.m = state.m or x.new(dfdx:size()):zero()
    -- Exponential moving average of squared gradient values
    state.v = state.v or x.new(dfdx:size()):zero()
    -- A tmp tensor to hold the sqrt(v) + epsilon
    state.denom = state.denom or x.new(dfdx:size()):zero()

    state.t = state.t + 1
    
    -- Decay the first and second moment running average coefficient
    state.m:mul(beta1):add(1 - beta1, dfdx)
    state.v:mul(beta2):addcmul(1 - beta2, dfdx, dfdx)

    state.denom:copy(state.v):sqrt():add(epsilon)

    local biasCorrection1 = 1 - beta1^state.t
    local biasCorrection2 = 1 - beta2^state.t
    local stepSize = lr * math.sqrt(biasCorrection2) / biasCorrection1
    -- (3) update x
    x:addcdiv(-stepSize, state.m, state.denom)

    return x
end
function rmsprop(x, dfdx, config, state)
    --[[ 
        This is rmsprop implementation based on the torch code,
        just changed a bit to reflect 
        our needs here.
        Source: https://github.com/torch/optim/blob/master/rmsprop.lua
    ]]
    -- (0) get/update state
    local config = config or {}
    local state = state or config
    local lr = config.learningRate or 1e-2
    local alpha = config.alpha or 0.99
    local epsilon = config.epsilon or 1e-8
    local wd = config.weightDecay or 0

    -- (1) weight decay
    if wd ~= 0 then
       dfdx:add(wd, x)
    end

    -- (3) initialize mean square values and square gradient storage
   if not state.m then
      state.m = torch.Tensor():typeAs(x):resizeAs(dfdx):fill(1)
      state.tmp = torch.Tensor():typeAs(x):resizeAs(dfdx)
   end

   -- (4) calculate new (leaky) mean squared values
   state.m:mul(alpha)
   state.m:addcmul(1.0 - alpha, dfdx, dfdx)

    -- (5) perform update
   state.tmp:sqrt(state.m):add(epsilon)
   x:addcdiv(-lr, dfdx, state.tmp)

   return x
 end
function adagrad(x, dfdx, config, state)
    --[[ 
        This is adagrad implementation based on the torch code,
        just changed a bit to reflect 
        our needs here.
        Source: https://github.com/torch/optim/blob/master/adagrad.lua
    ]]
    -- (0) get/update state
   if config == nil and state == nil then
      print('no state table, ADAGRAD initializing')
   end
   local config = config or {}
   local state = state or config
   local lr = config.learningRate or 1e-3
   local lrd = config.learningRateDecay or 0
   local wd = config.weightDecay or 0
   state.evalCounter = state.evalCounter or 0
   local nevals = state.evalCounter

   -- (1) weight decay with a single parameter
   if wd ~= 0 then
       dfdx:add(wd, x)
   end

   -- (2) learning rate decay (annealing)
   local clr = lr / (1 + nevals * lrd)

   -- (3) parameter update with single or individual learning rates
   if not state.paramVariance then
      state.paramVariance = torch.Tensor():typeAs(x):resizeAs(dfdx):zero()
      state.paramStd = torch.Tensor():typeAs(x):resizeAs(dfdx)
   end
   state.paramVariance:addcmul(1,dfdx,dfdx)
   state.paramStd:resizeAs(state.paramVariance):copy(state.paramVariance):sqrt()
   x:addcdiv(-clr, dfdx,state.paramStd:add(1e-10))

   -- (4) update evaluation counter
   state.evalCounter = state.evalCounter + 1

   return x
end

