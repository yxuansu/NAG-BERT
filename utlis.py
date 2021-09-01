import torch
import numpy as np
from dataclass import PAD, UNK, SOS, EOS

def map_text(batch_greedy_result, data):
    '''
        batch_greedy_result : torch Tensor
        vocab : vocabulary class
    '''

    padding_idx = data.tgt_vocab.padding_idx
    eos_idx = data.tgt_vocab.eos_idx
    unk_idx = data.tgt_vocab.unk_idx
    vocab = data.tgt_vocab

    batch_result = batch_greedy_result.cpu().detach().numpy()

    result = []
    for one_result in batch_result:
        one_res = []
        for one_idx in one_result:
            one_idx = int(one_idx)
            if one_idx == padding_idx:
                continue
            elif one_idx == eos_idx:
                break
            else:
                one_token = vocab.idx_token_dict[one_idx]
                one_res.append(one_token)
        one_res_text = ' '.join(one_res)
        result.append(one_res_text)
    return result

def get_article_ref_text(batch_greedy_result, data):
    batch_result = batch_greedy_result.cpu().detach().numpy()

    result = []
    for one_result in batch_result:
        one_text_list = ' '.join(data.vocab.convert_ids_to_tokens(one_result))
        one_text = ' '
        for token in one_text_list:
            if token == '[PAD]':
                pass
            else:
                one_text += token + ' '
        result.append(one_text.strip())
    return result

def get_title_ref_text(batch_greedy_result, data):
    '''
        batch_greedy_result : torch Tensor
        vocab : vocabulary class
    '''

    padding_idx = data.tgt_vocab.padding_idx
    eos_idx = data.tgt_vocab.eos_idx
    unk_idx = data.tgt_vocab.unk_idx
    vocab = data.tgt_vocab

    batch_result = batch_greedy_result.cpu().detach().numpy()

    result = []
    for one_result in batch_result:
        one_res = []
        for one_idx in one_result:
            one_idx = int(one_idx)
            if one_idx == padding_idx:
                continue
            #elif one_idx == eos_idx:
            #    break
            else:
                one_token = vocab.idx_token_dict[one_idx]
                one_res.append(one_token)
        one_res_text = ' '.join(one_res)
        result.append(one_res_text)
    return result

