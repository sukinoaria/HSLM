# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2017-10-17 16:47:32
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2018-03-30 16:20:07

import torch
import threading
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from basic.crf import CRF
from basic.wcrf import WeightedCRF
from basic.attention import Attention
from basic.wordEmbedding import WordEmbedding
from basic.wordHiddenRep import WordHiddenRep




class opinionMining(nn.Module):
    def __init__(self, args, data):
        super(opinionMining, self).__init__()
        print("build network...")
        self.gpu = args.ifgpu
        self.asp_label_size = data.asp_label_alphabet.size()
        self.opi_label_size = data.opi_label_alphabet.size()
        self.polar_size = data.polar_alphabet.size()

        self.asp_label_dim = args.asp_label_dim

        self.encoder_dim = args.encoder_dim

        # self.asp_key_dim = 200
        # self.asp_nonkey_dim = 100

        self.opinion_input_dim = args.opinion_input_dim
        self.opinion_fusion_dim = 300
        # self.polar_hidden_dim = 200


        # buliding model
        self.wordRepEmbedding = WordEmbedding(args, data)
        self.aspEmbedding = nn.Embedding(self.asp_label_size, self.asp_label_dim)
        self.aspEmbedding.weight.data.copy_(
            torch.from_numpy(self.random_embedding(self.asp_label_size, self.asp_label_dim)))

        # LSTM
        self.encoder = WordHiddenRep(args)

        # aspect tagging
        self.aspect2Tag = nn.Linear(self.encoder_dim, self.asp_label_size)
        #self.asp_crf = CRF(self.asp_label_size, self.gpu)

        # add nonlinear trans from asp feature to opinion feature
        self.asp_linear = nn.Linear(self.encoder_dim + self.asp_label_dim, self.opinion_input_dim)


        #wA+wB
        self.asp_key_linear = nn.Linear(self.encoder_dim+self.asp_label_dim,self.opinion_input_dim)
        self.asp_nonkey_linear = nn.Linear(self.encoder_dim,self.opinion_input_dim)

        # aspect tagging
        # get opinion feature , use opinion tag and sequence output , then use a linear
        self.opi_linear = nn.Linear(self.encoder_dim,self.opinion_fusion_dim)


        self.opinion2Tag = nn.Linear(self.opinion_input_dim, self.opi_label_size)
        #self.opinion2Tag = nn.Linear(self.encoder_dim + self.opinion_input_dim, self.opi_label_size)
        #self.opi_crf = WeightedCRF(self.opi_label_size, self.gpu)


        # ablation 1 only use seq out predict polarity

        # self.attn = Attention(args.ifgpu, self.encoder_dim, self.encoder_dim, args.head_dim, args.num_head)
        # self.classification = nn.Linear(self.encoder_dim + args.head_dim * args.num_head, self.polar_size)

        # ablation 2 only use asp out predict polarity
        #
        # self.attn = Attention(args.ifgpu, self.opinion_input_dim, self.encoder_dim, args.head_dim, args.num_head)
        # self.classification = nn.Linear(self.opinion_input_dim + args.head_dim * args.num_head, self.polar_size)

        # ablation 3 concat seq asp opi, not use attention

        #self.classification = nn.Linear(self.opinion_input_dim*2 + self.encoder_dim , self.polar_size)

        #self.classification = nn.Linear(self.opinion_input_dim + self.encoder_dim*2, self.polar_size)
        self.classification = nn.Linear(self.opinion_input_dim + self.encoder_dim +self.opinion_fusion_dim, self.polar_size)

        # ablation 4 seq out -> linear

        # self.classification = nn.Linear(self.encoder_dim, self.polar_size)

        # ablation 5 seq out; asp out -> linear

        # self.classification = nn.Linear(self.opinion_input_dim  + self.encoder_dim, self.polar_size)


        # ablation 6 concat asp opi -> linear

        #self.classification = nn.Linear(self.opinion_input_dim * 2, self.polar_size)
        #self.classification = nn.Linear(self.opinion_input_dim + self.encoder_dim, self.polar_size)
        #self.classification = nn.Linear(self.opinion_input_dim + self.opinion_fusion_dim, self.polar_size)


        # ablation 7 asp -> linear
        #self.classification = nn.Linear(self.opinion_input_dim, self.polar_size)

        # full model
        #
        self.attn = Attention(args.ifgpu, self.opinion_fusion_dim+self.opinion_input_dim, self.encoder_dim, args.head_dim,args.num_head)
        self.classification = nn.Linear(self.opinion_fusion_dim+self.opinion_input_dim+args.head_dim*args.num_head,self.polar_size)

        # Dropout
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, all_input_ids, input_length, input_recover, all_input_mask, all_char_ids, char_length,
                char_recover, char_mask):

        batch_size = all_input_ids.size(0)
        seq_len = all_input_ids.size(1)

        #### embedding lookup ####
        wordEmbedding = self.wordRepEmbedding(all_input_ids, all_char_ids, char_length, char_recover)
        embedding_mask = all_input_mask.view(batch_size, seq_len, 1).repeat(1, 1, wordEmbedding.size(2))
        wordEmbedding = wordEmbedding * (embedding_mask.float())

        #### LSTM embedding
        sequence_output = self.encoder(wordEmbedding, input_length)

        # aspect level result
        asp_out = self.aspect2Tag(sequence_output)
        _, asp_tag_seq = torch.max(asp_out.view(batch_size * seq_len, -1), 1)
        asp_tag_seq = asp_tag_seq.view(batch_size,seq_len)
        #asp_scores, asp_tag_seq = self.asp_crf._viterbi_decode(asp_out, all_input_mask.byte())

        # build opinion level features
        asp_features, opi_input = self.aspect_feature_fusion(sequence_output, asp_tag_seq)
        # asp_features = self.dropout(asp_features)
        # opi_input = self.dropout(opi_input)
        # opinion level tagging
        opi_out = self.opinion2Tag(opi_input).view(batch_size * seq_len, seq_len, -1)

        opi_mask = all_input_mask.unsqueeze(1).repeat(1, seq_len, 1).view(batch_size * seq_len, -1)
        maskTemp2 = all_input_mask.unsqueeze(2).repeat(1,1,seq_len).view(batch_size*seq_len,-1)
        opi_mask = opi_mask * maskTemp2

        #get opinion tag by torch max
        opi_out = opi_out.view(batch_size * seq_len * seq_len, -1)
        _, opi_tag_seq = torch.max(opi_out, 1)
        opi_tag_seq = opi_tag_seq.view(batch_size, seq_len,seq_len)
        ## filter padded position with zero
        opi_tag_seq = opi_mask.view(batch_size,seq_len,seq_len).long() * opi_tag_seq

        # get polar input features
        opi_features = self.opinion_feature_fusion(sequence_output, opi_tag_seq.view(batch_size*seq_len,seq_len))

        # as aspect feature fusion use add
        #polar_features = asp_features + opi_features
        #polar_features = torch.cat((asp_features, opi_features), dim=-1)

        # ablation 1 only use seq out predict polarity
        #
        # attn_context = self.attn.forward(sequence_output,sequence_output)
        # polar_features = torch.cat([sequence_output, attn_context], dim=-1)

        # ablation 2 only use asp out predict polarity

        # attn_context = self.attn.forward(asp_features, sequence_output)
        # polar_features = torch.cat([asp_features, attn_context], dim=-1)

        # ablation 3 concat seq asp opi, not use attention

        #polar_features = torch.cat([sequence_output,asp_features, opi_features], dim=-1)

        # ablation 4 seq out -> linear
        #polar_features = sequence_output

        # ablation 5 seq out; asp out -> linear
        #polar_features = torch.cat([sequence_output, asp_features], dim=-1)

        # ablation 6 concat asp opi -> linear
        #polar_features = torch.cat([opi_features, asp_features], dim=-1)

        # ablation 7 asp -> linear
        #polar_features = asp_features

        # full model

        polar_features = torch.cat([asp_features, opi_features], dim=-1)
        attn_context = self.attn.forward(polar_features, sequence_output)
        polar_features = torch.cat([polar_features, attn_context], dim=-1)


        probs = self.classification(polar_features)
        preds = torch.argmax(probs, dim=2)


        return asp_tag_seq, opi_tag_seq,preds, asp_out, opi_out,probs


    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def nll_loss(self, all_input_ids, input_length, input_recover, all_input_mask, all_char_ids,
                 char_length, char_recover, char_mask, asp_labels, opi_labels, polars):

        batch_size = all_input_ids.size(0)
        seq_len = all_input_ids.size(1)

        asp_tag_seq, opi_tag_seq,preds, asp_out, opi_out ,probs= self.forward(all_input_ids, input_length,
                                                                                input_recover,
                                                                                all_input_mask, all_char_ids,
                                                                                char_length, char_recover, char_mask)

        # aspect tagging loss
        # aspect_loss = self.asp_crf.neg_log_likelihood_loss(asp_out, all_input_mask.byte(), asp_labels) #/ (batch_size)
        asp_loss = nn.NLLLoss(ignore_index=0, size_average=False)
        asp_out = asp_out.view(batch_size * seq_len, -1)
        score = F.log_softmax(asp_out, 1)
        aspect_loss = asp_loss(score, asp_labels.view(-1))
        # opinion tagging loss
        # select non padding lines
        opinion_index = all_input_mask.view(batch_size * seq_len).nonzero().squeeze(1).long()
        opi_labels = opi_labels.view(batch_size * seq_len, -1)[opinion_index]
        opi_out = opi_out.view(batch_size * seq_len, -1)[opinion_index].view(-1,4)

        # weight matrix generate  [PAD,O,B,I]
        weight = torch.FloatTensor([1, 1, 1.0, 1.0])
        if self.gpu:
            weight = weight.cuda()
        opi_loss = nn.NLLLoss(weight=weight, ignore_index=0,size_average=False)
        score = F.log_softmax(opi_out, 1)
        opinion_loss = opi_loss(score, opi_labels.view(-1))

        # weight = torch.FloatTensor([0.01, 0.1, 1.0, 1.0])
        # if self.gpu:
        #     weight = weight.cuda()
        #
        # opi_loss = nn.NLLLoss(weight=weight, ignore_index=0, size_average=False)
        #
        # opi_out = opi_out.view(batch_size * seq_len * seq_len, -1)
        # score = F.log_softmax(opi_out, 1)
        # opinion_loss = opi_loss(score, opi_labels.view(batch_size * seq_len * seq_len))

        #polar loss
        # mask padding polar results
        gold_polars = torch.masked_select(polars, all_input_mask.bool()).long()
        probs = torch.masked_select(probs, all_input_mask.unsqueeze(2).repeat(1, 1, self.polar_size).bool()).view(-1,
                                                                                                                  self.polar_size)
        weight = torch.FloatTensor([0.1, 1.0, 1.0, 1.0])
        if self.gpu:
            weight = weight.cuda()
        polar_loss_func = nn.CrossEntropyLoss(weight=weight, size_average=False)
        polar_loss = polar_loss_func(probs, gold_polars)#/batch_size

        return asp_tag_seq, opi_tag_seq, preds, aspect_loss, opinion_loss, polar_loss



    def aspect_feature_fusion(self, sequence_output, tag_seq):
        batch_size = tag_seq.size(0)
        seq_len = tag_seq.size(1)

        threads = []

        fusion_mask = torch.stack([torch.eye(seq_len) for _ in range(batch_size)])
        # variant 1 : zero mask
        #fusion_mask = torch.zeros((batch_size,seq_len,seq_len))

        if self.gpu:
            fusion_mask = fusion_mask.cuda()
        for i in range(batch_size):
            t = threading.Thread(target=self.aspect_boundary_mask, args=(i, tag_seq[i, :], fusion_mask))
            threads.append(t)
        for i in range(batch_size):
            threads[i].start()
        for i in range(batch_size):
            threads[i].join()

        # divide line sum
        fusion_mask = fusion_mask.reshape(batch_size * seq_len, seq_len)
        fusion_sum = fusion_mask.sum(dim=1).view(batch_size * seq_len, 1).repeat(1, seq_len)
        fusion_mask = fusion_mask / fusion_sum

        # set nan to zero
        fusion_mask[fusion_mask != fusion_mask] = 0

        fusion_mask = fusion_mask.view(batch_size, seq_len, seq_len)

        # fusion same entity hidden state
        # the results only is the merged aspect representations
        sequence_output_1 = fusion_mask.bmm(sequence_output)

        # 先合并了label的表示之后再进行不同单词间表示的合并
        asp_tag_embeddings = self.aspEmbedding(tag_seq)
        asp_features = torch.cat((sequence_output_1, asp_tag_embeddings), dim=2)

        asp_key_features = self.asp_key_linear(asp_features)
        asp_nonkey_features = self.asp_nonkey_linear(sequence_output)

        # variant 2 : add fusion
        opi_input = asp_key_features.unsqueeze(1).repeat(1, seq_len, 1, 1) +asp_nonkey_features.unsqueeze(2).repeat(1, 1, seq_len,
                                                                                                        1)
        opi_input = F.relu(opi_input.permute(0,2,1,3))

        # opi_input = torch.cat(
        #     (asp_key_features.unsqueeze(1).repeat(1, seq_len, 1, 1), asp_nonkey_features.unsqueeze(2).repeat(1, 1, seq_len, 1)),
        #     dim=-1).permute(0, 2, 1, 3)
        # opi_input = torch.tanh(opi_input)
        return asp_key_features, opi_input
        #return asp_features, opi_input

    def aspect_boundary_mask(self, idx, tag_seq, fusion_mask):
        # don't consider the entity which starts with "I-X"
        tag_seq = tag_seq.cpu().data.numpy()

        start = -1
        FIND = False
        for i, tag in enumerate(tag_seq):
            if tag == 2:
                if not FIND:
                    FIND = True
                    start = i
                else:  # 2个B相连
                    FIND = True
                    fusion_mask[idx, start:i, start:i] = 1
                    start = i
            elif tag < 2:
                if FIND:
                    FIND = False
                    fusion_mask[idx, start:i, start:i] = 1
            else:
                continue
        if FIND:  # last one solo aspect
            fusion_mask[idx, i:i + 1, i:i + 1] = 1

        return fusion_mask

    def opinion_feature_fusion(self, features, tag_seq):
        seq_len = tag_seq.size(1)
        batch_size = tag_seq.size(0) // seq_len

        threads = []
        fusion_mask = torch.stack([torch.eye(seq_len) for _ in range(batch_size)]).view(batch_size * seq_len, seq_len)
        #fusion_mask = torch.zeros((batch_size,seq_len,seq_len)).view(batch_size * seq_len, seq_len)
        if self.gpu:
            fusion_mask = fusion_mask.cuda()
        for i in range(batch_size * seq_len):
            t = threading.Thread(target=self.opinion_boundary_mask, args=(i, tag_seq[i, :], fusion_mask))
            threads.append(t)
        for i in range(batch_size * seq_len):
            threads[i].start()
        for i in range(batch_size * seq_len):
            threads[i].join()

        # divide line sum
        fusion_sum = fusion_mask.sum(dim=1).view(batch_size * seq_len, 1).repeat(1, seq_len)
        fusion_mask = fusion_mask / fusion_sum

        # set nan to zero
        fusion_mask[fusion_mask != fusion_mask] = 0

        fusion_mask = fusion_mask.view(batch_size, seq_len, seq_len)

        # fusion same entity hidden state
        opinion_out = fusion_mask.bmm(features)

        # fusion opinion label embedding

        opinion_out = self.opi_linear(opinion_out)
        #opinion_out = F.relu(self.opi_linear(opinion_out))

        return opinion_out

    def opinion_boundary_mask(self, idx, tag_seq, fusion_mask):
        # don't consider the entity which starts with "I-X"
        tag_seq = tag_seq.cpu().data.numpy()

        start = -1
        FIND = False
        for i, tag in enumerate(tag_seq):
            if tag == 2:
                if not FIND:
                    FIND = True
                    start = i
                else:  # 2个B相连
                    FIND = True
                    fusion_mask[idx, start:i] = 1
                    start = i
            elif tag < 2:
                if FIND:
                    FIND = False
                    fusion_mask[idx, start:i] = 1
            else:
                continue
        if FIND:  # last one solo aspect
            fusion_mask[idx, i:i + 1] = 1

        return fusion_mask
