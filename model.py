import torch
from torch import nn
import torch.nn.functional as F
import random
from dynamic_crf_layer import *

class TopLayer(nn.Module):
    def __init__(self, vocab_size, embed_dim, crf_low_rank, crf_beam_size, dropout, padding_idx):
        super(TopLayer, self).__init__()

        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.padding_idx = padding_idx

        self.crf_layer = DynamicCRF(num_embedding = vocab_size, low_rank = crf_low_rank, 
                                    beam_size = crf_beam_size)

        self.one_more_layer_norm = nn.LayerNorm(embed_dim)
        self.tgt_word_prj = nn.Linear(self.embed_dim, self.vocab_size)

    def forward(self, src_representation, src_input, tgt_input, is_training):
        '''
            src_representation : bsz x seqlen x embed_dim
            src_input : bsz x seqlen
            tgt_input : bsz x seqlen
        '''
        assert src_input.size() == tgt_input.size()

        src_input = src_input.transpose(0, 1) # src_len x bsz
        seqlen, bsz = src_input.size()

        src_representation = F.dropout(src_representation, p=self.dropout, training=is_training)
        src_representation = src_representation.transpose(0, 1) # seqlen x bsz x embed_dim

        src = src_representation

        emissions = self.tgt_word_prj(src.contiguous().view(-1, self.embed_dim)).view(seqlen, bsz, self.vocab_size)
        log_probs = torch.log_softmax(emissions, -1)
        assert log_probs.size() == torch.Size([seqlen, bsz, self.vocab_size])

        emissions = emissions.transpose(0, 1) # [bsz x src_len x vocab_size]
        emission_mask = ~tgt_input.eq(self.padding_idx) # [bsz x src_len]
        batch_crf_loss = -1 * self.crf_layer(emissions, tgt_input, emission_mask) # [bsz]
        assert batch_crf_loss.size() == torch.Size([bsz])
        return log_probs, batch_crf_loss

    def decoding(self, src_representation, src_input):
        '''
            src_representation : bsz x seqlen x embed_dim
            src_input : bsz x seqlen
            tgt_input : bsz x seqlen
        '''
        src_input = src_input.transpose(0, 1) # src_len x bsz
        seqlen, bsz = src_input.size()

        src_representation = src_representation.transpose(0, 1) # seqlen x bsz x embed_dim
        src = src_representation

        emissions = self.tgt_word_prj(src.contiguous().view(-1, self.embed_dim)).view(seqlen, bsz, self.vocab_size)

        emissions = emissions.transpose(0, 1) # [bsz, seqlen, vocab_size]
        _, finalized_tokens = self.crf_layer.forward_decoder(emissions)
        assert finalized_tokens.size() == torch.Size([bsz, seqlen])
        return finalized_tokens

    def length_ratio_decoding(self, src_representation, src_input, length_ratio):
        '''
            src_representation : 1 x seqlen x embed_dim
            src_input : 1 x seqlen
        '''
        src_input = src_input.transpose(0, 1) # src_len x bsz
        seqlen, bsz = src_input.size()

        src_representation = src_representation.transpose(0, 1) # seqlen x bsz x embed_dim
        src = src_representation

        emissions = self.tgt_word_prj(src.contiguous().view(-1, self.embed_dim)).view(seqlen, bsz, self.vocab_size)

        emissions = emissions.transpose(0, 1) # [bsz, seqlen, vocab_size]
        valid_len = int(seqlen * length_ratio) + 1
        valid_emissions = emissions[:, :valid_len+1,:]
        _, finalized_tokens = self.crf_layer.forward_decoder(valid_emissions)
        return finalized_tokens

class NAG_BERT(nn.Module):
    def __init__(self, bert_model, vocab_size, embed_dim, crf_low_rank, crf_beam_size, dropout, src_padding_idx, tgt_padding_idx):
        super(NAG_BERT, self).__init__()
        self.embed_dim = embed_dim
        self.bert_model = bert_model
        self.toplayer = TopLayer(vocab_size, embed_dim, crf_low_rank, crf_beam_size, dropout, tgt_padding_idx)
        self.src_padding_idx = src_padding_idx
        self.tgt_padding_idx = tgt_padding_idx

    def forward(self, src_input, tgt_input, is_training):
        '''
            src_input : bsz x seqlen
            tgt_input : bsz x seqlen 
        '''
        bsz, seqlen = src_input.size()
        src_mask = ~src_input.eq(self.src_padding_idx)
        src_representation, _ = self.bert_model(src_input, attention_mask = src_mask, 
                                                output_all_encoded_layers = False)
        assert src_representation.size() == torch.Size([bsz, seqlen, self.embed_dim])
        log_probs, batch_crf_loss = self.toplayer(src_representation, src_input, tgt_input, is_training)
        return log_probs, batch_crf_loss

    def decoding(self, src_input):
        src_mask = ~src_input.eq(self.src_padding_idx)
        src_representation, _ = self.bert_model.work(src_input, attention_mask = src_mask, 
                                                output_all_encoded_layers = False)
        finalized_tokens = self.toplayer.decoding(src_representation, src_input)
        return finalized_tokens


    def length_ratio_decoding(self, src_input, length_ratio):
        src_mask = ~src_input.eq(self.src_padding_idx)
        src_representation, _ = self.bert_model(src_input, attention_mask = src_mask, 
                                                output_all_encoded_layers = False)

        finalized_tokens = self.toplayer.length_ratio_decoding(src_representation, 
                            src_input, length_ratio)
        return finalized_tokens

    def length_ratio_decoding_no_dropout(self, src_input, length_ratio):
        src_mask = ~src_input.eq(self.src_padding_idx)
        src_representation, _ = self.bert_model.work(src_input, attention_mask = src_mask, 
                                                output_all_encoded_layers = False)

        finalized_tokens = self.toplayer.length_ratio_decoding(src_representation, 
                            src_input, length_ratio)
        return finalized_tokens


