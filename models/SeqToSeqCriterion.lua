-------------------------------------------------------------------------------
-- SeqToSeq model-aware Criterion
-------------------------------------------------------------------------------

local crit, parent = torch.class('nn.SeqToSeqCriterion', 'nn.Criterion')
function crit:__init(video_length)
  parent.__init(self)
  self.video_len =  video_length
end

--[[
input is a Tensor of size  L x B x  V+1 
where L is sequence length including frames + 1 + sentence_len
B is batch size
V is Vocab_size + 1
--]]
function crit:updateOutput(input, seq)
  self.gradInput:resizeAs(input):zero() -- reset to zeros
  local L, B, Mp1 = input:size(1), input:size(2), input:size(3)
  local D = seq:size(1)
  assert(D == L - self.video_len -1 , 'input Tensor should be video_length + 1 larger in time')

  local loss = 0
  local n = 0
  for b = 1, B do -- iterate over batches
    local first_time = true
    for t = 1 + self.video_len , L do -- iterate over seq time (ignore t= 1 + self.video_len, dummy forward for the videos)

      -- fetch the index of the next token in the sequence
      local target_index
      if t == L then -- This is the last token in the seqeunce, we are out of bounds of the index sequence: pad with null tokens
        target_index = 0
      else
        target_index = seq[{t - self.video_len, b}] -- t-self.video_len is correct, since we have videos at the begining
      end
      -- the first time we see null token as next index, actually want the model to predict the END token
      if target_index == 0 and first_time then
        target_index = Mp1
        first_time = false
      end

      -- if there is a non-null next token, enforce loss!
      if target_index ~= 0 then
        -- accumulate loss
        loss = loss - input[{ t, b, target_index }] -- log(p)
        self.gradInput[{ t, b, target_index }] = -1 --  -dp --> -1
        n = n + 1
      end

    end
  end
  self.output = loss / n -- normalize by number of predictions that were made
  self.gradInput:div(n)
  return self.output
end

function crit:updateGradInput(input, seq)
  return self.gradInput
end