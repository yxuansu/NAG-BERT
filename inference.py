import argparse
import datetime
import torch
import os
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
from model import NAG_BERT
from dataclass import Data, PAD, UNK, SEP, CLS, SOS, EOS
from utlis import map_text
from evaluation import write_results, get_rouge_scores

def init_bert_model(bert_path):
    bert_vocab = BertTokenizer.from_pretrained(bert_path)
    bert_model = BertModel.from_pretrained(bert_path)
    return bert_model, bert_vocab

def parse_config():
    parser = argparse.ArgumentParser()
    # model parameters path
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--length_ratio', type=float)
    parser.add_argument('--test_path', type=str)
    parser.add_argument('--tgt_vocab_f', type=str)
    parser.add_argument('--tgt_vocab_size', type=int, default=150000)
    parser.add_argument('--seq_max_len', type=int, default=128)
    parser.add_argument('--gpu_id', type=int, default=0)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_config()
    summary_dir = args.ckpt_path + r'./decoded_result/'
    try:
        os.stat(summary_dir)
    except:
        os.mkdir(summary_dir)

    # path to store reference summary
    model_dir = args.ckpt_path + r'./reference_result/'
    try:
        os.stat(model_dir)
    except:
        os.mkdir(model_dir)

    device = args.gpu_id
    print ('Loading Model...')
    bert_model, bert_vocab = init_bert_model(args.model_name)
    model_ckpt = torch.load(args.ckpt_path + r'./best_ckpt')
    model_args = model_ckpt['args']

    data = Data(args.test_path, args.test_path, bert_vocab, args.seq_max_len, args.tgt_vocab_f, args.tgt_vocab_size)
    dev_data_num = data.dev_num
    dev_step_num = int(dev_data_num)

    # model configuration
    vocab_size = data.tgt_vocab_size
    embed_dim, crf_low_rank, crf_beam_size = bert_model.config.hidden_size, model_args.crf_low_rank, model_args.crf_beam_size
    dropout = 0.

    model = NAG_BERT(bert_model, vocab_size, embed_dim, crf_low_rank, crf_beam_size, dropout, data.src_padding_idx, data.tgt_padding_idx)

    model_parameters = model_ckpt['model']
    model.load_state_dict(model_parameters)
    model = model.cuda(args.gpu_id)
    print ('Model Loaded.')

    reference = []
    hypothesis = []

    total_millisecond_cnt = 0.0
    total_token_cnt = 0
    token_instance_cnt = 0
    with torch.no_grad():
        for dev_step in range(dev_step_num):
            if dev_step % int(dev_step_num / 10) == 0:
                print ('%d instances have been processed' % dev_step)
                
            is_training = False
            dev_batch_src_inp, dev_batch_tgt, dev_batch_truth = data.get_next_batch(1, mode = 'dev')

            dev_batch_src_inp = torch.LongTensor(dev_batch_src_inp).cuda(device)
            dev_batch_tgt = torch.LongTensor(dev_batch_tgt).cuda(device)
            reference += dev_batch_truth 

            _, one_tgt_len = dev_batch_tgt.size()

            # evaluate BLEU score
            start = datetime.datetime.now()
            dev_batch_result = model.length_ratio_decoding(dev_batch_src_inp, args.length_ratio)
            end = datetime.datetime.now()
            one_time_diff = end - start
            one_millisecond_cnt = one_time_diff.total_seconds() * 1000
            total_millisecond_cnt += one_millisecond_cnt

            _, seqlen = dev_batch_result.size()
            total_token_cnt += seqlen
            token_instance_cnt += 1

            dev_batch_hypothesis = map_text(dev_batch_result, data)
            hypothesis += dev_batch_hypothesis

    write_results(summary_dir, hypothesis, mode = r'decoded')
    # write reference result
    write_results(model_dir, reference, mode = r'reference')
    # compute score
    rogue_1_score, rogue_2_score, rogue_l_score = get_rouge_scores(summary_dir, model_dir)
    print ('rogue 1 is %.5f, rogue 2 is %.5f, rogue l is %.5f' % (rogue_1_score, rogue_2_score, rogue_l_score))

    ave_token_time = round(total_millisecond_cnt / total_token_cnt, 3)
    ave_instance_time = round(total_millisecond_cnt / token_instance_cnt, 3)
    print ('Total Decoding Time is %.5f, Average token time is %.5f, instance time is %.5f' 
        % (total_millisecond_cnt, ave_token_time, ave_instance_time))


