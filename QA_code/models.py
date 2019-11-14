"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import torch
import torch.nn as nn
# from transformer import *
from encoder import EncoderBlock


class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """

    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob=0.):
        super(BiDAF, self).__init__()
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    char_vectors=char_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        # self.enc = layers.RNNEncoder(input_size=hidden_size,
        #                              hidden_size=hidden_size,
        #                              num_layers=1,
        #                              drop_prob=drop_prob)

        # self.transformer = make_model(word_vectors, drop_prob, hidden_size)

        self.emb_enc = EncoderBlock(conv_num=4, ch_num=64, k=7)

        self.att = layers.BiDAFAttention(hidden_size=hidden_size,
                                         drop_prob=drop_prob)

        # TODO
        self.mod = layers.RNNEncoder(input_size=4 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        # print("c_mask: ", c_mask.size())    # [64, 362]
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        # print("q_mask: ", q_mask.size())    # [64, 23]

        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs, cc_idxs)  # (batch_size, c_len, hidden_size)
        # c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
        # print("c_emb: ", c_emb.size())     # [64, 362, 128]

        q_emb = self.emb(qw_idxs, qc_idxs)  # (batch_size, q_len, hidden_size)
        # q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)
        # print("q_emb: ", q_emb.size())      # [64, 23, 128]

        # rotate since code because in transformer dimension 1 and 2 are reversed
        c_emb, q_emb = c_emb.transpose(1, 2), q_emb.transpose(1, 2)
        c_enc = self.emb_enc(c_emb, c_mask, 1, 1)  # (batch_size, c_len, hidden_size)
        # print("c_enc: ", c_enc.size())

        q_enc = self.emb_enc(q_emb, q_mask, 1, 1)  # (batch_size, q_len, hidden_size)
        # print("q_enc: ", q_enc.size())
        # print("c_enc2: ", c_enc2.size(), " q_enc2: ", q_enc1.size())
        c_enc, q_enc = c_enc.transpose(1, 2), q_enc.transpose(1, 2)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)  # (batch_size, c_len, 4 * hidden_size)
        # print("att: ", att.size())  # [64, 373, 1536]

        mod = self.mod(att, c_len)  # (batch_size, c_len, 2 * hidden_size)
        # print("mode: ", mod.size(), " c_mask: ", c_mask.size())  # [64, 373, 256]

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)
        # print(out.size())
        # return
        return out
