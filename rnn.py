"""
RNNs/language models etc
"""


import torch
import torch.nn.utils.rnn as rnn_utils
import torch.nn as nn


class RNNEncoder(nn.Module):
    """
    RNN Encoder - takes in onehot representations of tokens, rather than numeric
    """

    def __init__(self, embedding_module, hidden_size=100):
        super().__init__()
        self.embedding = embedding_module
        self.embedding_dim = embedding_module.embedding_dim
        self.hidden_size = hidden_size
        self.gru = nn.GRU(self.embedding_dim, hidden_size)

    def forward(self, seq, length):
        batch_size = seq.size(0)
        #print("batch_size: ", batch_size) #KK: 32
        #print("length: ", length) #KK: MPS-tensor of size 32 containing lengths up to 7
        
        #print("seq before sorting: ", seq.shape)

        if batch_size > 1:
            sorted_lengths, sorted_idx = torch.sort(length, descending=True)
            seq = seq[sorted_idx]
            
        #print("sorted_lengths: ", sorted_lengths)
        #print("sorte index: ", sorted_idx)
        #print("seq: ", seq)

        #print("seq after sorting and before reordering: ", seq.shape)
        # reorder from (B,L,D) to (L,B,D)
        seq = seq.transpose(0, 1)
        #print("seq after reordering: ", seq.shape)

        # embed your sequences
        embed_seq = seq @ self.embedding.weight
        
        #print("in rnn.py - seq: ", seq.type, "self.embedding.weight: ", self.embedding.weight.type, "embed_seq: ", embed_seq.shape)
        
        #print("sorted_lengths.data.tolist(): ", sorted_lengths.data.tolist())

        packed = rnn_utils.pack_padded_sequence(
            embed_seq,
            sorted_lengths.data.tolist() if batch_size > 1 else length.data.tolist(),
        )
        
        #print("packed: ", packed.data.shape)

        _, hidden = self.gru(packed)
        hidden = hidden[-1, ...] #KK: obtaining the last hidden state

        if batch_size > 1:
            _, reversed_idx = torch.sort(sorted_idx)
            hidden = hidden[reversed_idx]

        return hidden

    def reset_parameters(self):
        self.embedding.reset_parameters()
        self.gru.reset_parameters()
