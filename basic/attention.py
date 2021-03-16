# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class Attention(nn.Module):
    def __init__(self, ifgpu,q_dim,k_dim,head_dim,num_head):
        super(Attention, self).__init__()
        self.gpu = ifgpu

        self.q_input_dim = q_dim
        self.k_input_dim = k_dim

        self.head_dim = head_dim
        self.num_heads = num_head

        self.scaling = self.head_dim ** (-0.5)

        self.q_linear = nn.Linear(self.q_input_dim, self.num_heads * self.head_dim, bias=True)
        self.k_linear = nn.Linear(self.k_input_dim, self.num_heads * self.head_dim, bias=True)

        nn.init.xavier_uniform_(self.q_linear.weight)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.constant_(self.q_linear.bias, 0)
        nn.init.constant_(self.k_linear.bias, 0)


    def forward(self, query_input,key_input):
        """
            input:
                wordHidden: (batch_size, sent_len, word_hidden_dim)
                target_a: (batch_size, sent_len, word_hidden_dim)
            output:
                Variable(batch_size, sent_len, hidden_dim)
        """
        # bmm(linear(query),key) h_tWh_i
        # Wtanh(w*q+w*k+b)
        batchSize = query_input.size(0)
        seqLen = query_input.size(1)

        query = query_input.transpose(0,1) # seq x batchsize x hd
        key = key_input.transpose(0,1) # seq x batchsize x hd

        q = self.q_linear(query) # seq x batchsize x num_heads*head_dim
        k = self.k_linear(key)

        q = q.contiguous().view(seqLen, batchSize * self.num_heads, self.head_dim).transpose(0, 1) # batchsize*num_heads x seqlen x head_dim
        k = k.contiguous().view(seqLen, batchSize * self.num_heads, self.head_dim).transpose(0, 1)
        q *= self.scaling

        attn_weights = torch.bmm(q, k.transpose(1, 2)).view(batchSize, self.num_heads, seqLen, seqLen) # batchsize x num_heads x seqlen x seqlen
        attn_weights = F.softmax(attn_weights, dim=3) # batchsize x num_heads x seqlen x seqlen

        # attn_weight mult v
        attned = attn_weights.view(batchSize * self.num_heads, seqLen, -1).bmm(k)

        # reshape attened features concat multi head result
        attned = attned.view(batchSize,self.num_heads,seqLen,self.head_dim)

        # k = k.view(batchSize,self.num_heads,seqLen,self.head_dim)
        #
        # attned = k + attned

        attned = attned.permute(0,2,1,3).reshape(batchSize,seqLen,self.num_heads*self.head_dim)
        return attned