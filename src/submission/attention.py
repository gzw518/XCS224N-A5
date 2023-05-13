import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    I believe I could have just used torch.nn.MultiheadAttention but their documentation
    is all but absent and code ugly so I don't trust it, rolling my own here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        # (B x T x C) is of dimension (batch x block_size x n_embd) which is (batch x l x d) in the handout.
        # nh should be number_of_heads, and hs would then stand for n_embed (or "dimensionality" d in the handout) per head

        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, -1e10) # todo: just use float('-inf') instead?
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class SynthesizerAttention(nn.Module):
    """
    A synthesizer multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # MLP Params
        self.w1 = nn.Linear(config.n_embd, config.n_embd)
        self.w2 = nn.Parameter(torch.zeros(config.n_embd // config.n_head,
            config.block_size-1))
        self.b2 = nn.Parameter(torch.zeros(config.block_size-1))
        # value projection
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in
        #     the input sequence
        self.register_buffer("mask", torch.tril(
            torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.block_size = config.block_size

        nn.init.uniform_(self.w2,-0.001,0.001)

    def forward(self, x, layer_past=None):

        ### TODO:
        ### [part g]: Write your SynthesizerAttention below.
        ###   Do not modify __init__().
        ### Hints:
        ###   - Paste over the CausalSelfAttention above and modify it minimally.
        ###   - Consider especially the parameters self.w1, self.w2 and self.b2.
        ###       How do these map to the matrices in the handout?

        ### START CODE HERE

        B, T, C = x.size()      # B: batch size, T: block size, C: embeding dimension  (batch x block_size x n_embd)
        hs = C // self.n_head   # hs: would then stand for n_embed (or "dimensionality" d in the handout) per head
        nh = self.n_head        # nh: should be number_of_heads,

        v = self.value(x).view(B, T, nh, hs).transpose(1, 2)  # (B, nh, T, hs)
        # x linear() with w1, and linear() function has default bias as b1
        x = self.w1(x).view(B, T, nh, hs).transpose(1, 2)     # (B, nh, T, hs)

        print("self.block_size:",self.block_size)
        print("x size:",x.size())
        print("v size:",v.size())

        # F.relu(x) is (B, nh, T, hs) and self.w2 (hs, BlockSize-1), so (B, nh, T, hs) x (hs, BlockSize-1) -> (B, nh, T, BlockSize-1), and b2 is (BlockSize-1)
        att = torch.matmul(F.relu(x), self.w2) + self.b2
        print("att = torch.matmul(F.relu(x), self.w2) + self.b2, then att size:",att.size())

        #
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, -1e10) # todo: just use float('-inf') instead?

        att = F.softmax(att, dim=-1)       # dim = -1 means use the last dimension (size is T-1) to softmax
        print("att = F.softmax(att, dim=-1), then att size:", att.size())

        att = self.attn_drop(att)

        print("B, T, C, hs, nh:", B,T,C,hs,nh)
        print("att size:", att.size())
        print("v size:", v.size())

        y = torch.matmul(att, v)   # (B, nh, T, T-1) x (B, nh, hs, T) -> (B, nh, T, hs)
        print("y = torch.matmul(att, v), then y size:", y.size())

        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        print("y = y.transpose(1, 2).contiguous().view(B, T, C), then y size:", y.size())

        # output projection
        y = self.resid_drop(self.proj(y))

        #print("*** Finally the y value: ",y)

        #
        # # SynthesizerAttention
        # #X = self.w1(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        # X = x
        # A = self.w1(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # # self.w2 = nn.Parameter(torch.zeros(config.n_embd // config.n_head,config.block_size-1))
        # # B = self.w2(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # V = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        #
        # # SynthesizerAttention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # #att = nn.ReLU(X @ A.transpose(-2,-1) + self.b2) @ self.w2 + self.b2
        #
        # XA = x.matmul(A) + self.w2
        #
        #
        # att = att.masked_fill(self.mask[:,:,:T,:T] == 0, -1e10) # todo: just use float('-inf') instead?
        # att = F.softmax(att, dim=-1)
        # att = self.attn_drop(att)
        # y = att @ V # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        #
        # # output projection
        # y = self.resid_drop(self.proj(y))

        return y        
        
        ### END CODE HERE

        raise NotImplementedError



##########################
# for my debug
##########################

class sample_GPTConfig:
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    synthesizer = False

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)



if __name__ == '__main__':

    torch.manual_seed(0)

    # # Instantiate a namedtuple to mimic a configuration object
    # Config = namedtuple("Config", ["n_embd", "n_head", "block_size",
    #                                "attn_pdrop", "resid_pdrop"])
    # config = Config(n_embd=512, n_head=8, block_size=12,
    #                 attn_pdrop=0.1, resid_pdrop=0.1)

    config = sample_GPTConfig(5, 8, n_layer=1, n_head=3, n_embd=6)

    # Instantiate the SynthesizerAttention class
    synth_attn = SynthesizerAttention(config)

    # Create a random input tensor of shape (batch_size, block_size, n_embd)
    x = torch.randn(6, config.block_size, config.n_embd)

    # Call the forward method and print the output
    output = synth_attn.forward(x)
    print(f"Output shape: {output.shape}")

