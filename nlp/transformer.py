import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax
import copy

class Transformer(nn.Module):
    """
    An encoder-decorder architecture with self-attention mechanism. Most codes are from https://nlp.seas.harvard.edu/annotated-transformer/.
    This is just a demo for self-learning. Please use the original code for research purposes.

    Summary:
    •   Self-attention mechanism allows parallelization
    •   Encoder sub-layer: Multi-head Atention + FFN
    •   Decoder sub-layer: Masked Multi-head Attention (weight = 0) + Multi-head Attention + FFN
    •   LayerNorm (local, [C, H, W]) vs BatchNorm (global, [N, H, W])
    •   Scale the dot-product attention to avoid large dot-product values (i.e. extremely small gradients) when d_k is large
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
        Q: (n x d_k), K: (m x d_k), V: (m x d_v), Attention: (n x d_v)
    •   FFN(x) = ReLU(xW_1 + b_1)W_2 + b_2
        W_1: d_model = 512 -> d_ff = 2048, W_2: d_ff = 2048 -> d_model = 512

    References:
    https://arxiv.org/pdf/1706.03762.pdf
    https://nlp.seas.harvard.edu/annotated-transformer
    https://github.com/jadore801120/attention-is-all-you-need-pytorch
    https://nlp.cs.hku.hk/comp3314-spring2023/tutorial_3.ipynb
    """

    # Originally the "make_model" function
    def __init__(self, src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
        super(Transformer, self).__init__()

        c = copy.deepcopy
        attn = MultiHeadAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)

        self.model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
            nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
            nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
            Generator(d_model, tgt_vocab),
        )

        # Initialize parameters with Glorot / fan_avg
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

class Embeddings(nn.Module):
    """
    Word embeddings to convert tokens to unique vector representations.
    """

    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        # Multiply weights by sqrt(d_model)
        return self.lut(x) * np.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    """
    Sinusoidal encoding. Injects positional information into the sequence.
    """

    def __init__(self, d_model, dropout, max_len=200):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        # PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)

        # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        # Register PE as a buffer (i.e., not a parameter)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism. Used in encoder-decoder attention layers.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0, "d_model must be divisible by h"
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.attn = None
        self.dropout = nn.Dropout(dropout)

    def attention(self, query, key, value, mask=None, dropout=None):
        d_k = query.size(1)

        # Compute scaled dot-product attention: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
        scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = scores.softmax(dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        n_batch = query.size(0)

        # Linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(n_batch, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # Apply self-attention mechanism to all projected vectors in batch
        x, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # "Concat" using a view and apply a final linear
        x = (x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k))

        # Release memory
        del query
        del key
        del value

        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    """
    A 2-layer position-wise feed forward network (FFN).
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))

class LayerNorm(nn.Module):
    """
    A layer normalization module. Output of each sub-layer = LayerNorm(x + Sublayer(x)).
    """

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class Sublayer(nn.Module):
    """
    A residual connection followed by a layer normalization.
    """

    def __init__(self, size, dropout):
        super(Sublayer, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    """
    A layer of encoder stack. Consists of multi-head attention and feed forward network (FFN) sub-layers.
    """

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayers = nn.ModuleList([Sublayer(size, dropout) for _ in range(2)])
        self.size = size

    def forward(self, x, mask):
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayers[1](x, self.feed_forward)

class Encoder(nn.Module):
    """
    A stack of N encoder layers.
    """

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([layer for _ in range(N)])
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)

class DecoderLayer(nn.Module):
    """
    A layer of decoder stack. Consists of masked multi-head attention, multi-head attention and feed forward network (FFN) sub-layers.
    """

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayers = nn.ModuleList([Sublayer(size, dropout) for _ in range(3)])

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayers[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayers[2](x, self.feed_forward)

class Decoder(nn.Module):
    """
    A stack of N decoder layers.
    """

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([layer for _ in range(N)])
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)

        return self.norm(x)

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        # Consume previously generated symbols as additional input when generating the next
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

class Generator(nn.Module):
    """
    Define standard linear + softmax generation step.
    """

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)
