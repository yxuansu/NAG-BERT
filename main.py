import sys
import os
import operator
from operator import itemgetter
import torch
from torch import nn
import torch.nn.functional as F
import random
#from dataclass import Data
import numpy as np
import argparse

from dataclass import Data
from model import NAG_BERT

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel

from utlis import map_text, get_article_ref_text, get_title_ref_text
from evaluation import write_results, get_rouge_scores

def init_bert_model(bert_path):
    bert_vocab = BertTokenizer.from_pretrained(bert_path)
    bert_model = BertModel.from_pretrained(bert_path)
    return bert_model, bert_vocab

def compute_lr(step, warmup):
    num1 = step ** (-0.5)
    num2 = step * (warmup ** (-1.5))
    return min(num1, num2)

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def create_directory(path):
    import os
    if os.path.exists(path):
        os.system('rm -r ' + path)
    os.makedirs(path, exist_ok=True)

def rm_tmp_files():
    folder_name_list = os.listdir(r'/tmp')
    for name in folder_name_list:
        if name.startswith('tmp'):
            path = r'/tmp/' + name
            os.system('rm -r ' + path)

def parse_config():
    parser = argparse.ArgumentParser()
    # data configuration
    parser.add_argument('--model_name', type=str, default='bert-base-uncased', help="e.g. bert-base-uncased")
    parser.add_argument('--train_data', type=str)
    parser.add_argument('--dev_data', type=str)
    parser.add_argument('--tgt_vocab_f', type=str)
    parser.add_argument('--tgt_vocab_size', type=int, default=150000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seq_max_len', type=int, default=128)

    # CRF layer configuration
    parser.add_argument('--crf_low_rank', type=int, default=64)
    parser.add_argument('--crf_beam_size', type=int, default=256)

    # training configuration
    parser.add_argument('--nll_loss_weight', type=float, default=0.5)
    parser.add_argument('--dropout',type=float, default=0.2)
    parser.add_argument('--number_epoch', type = int, default=50)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--print_every', type=int, default=400)
    parser.add_argument('--eval_every', type=int, default=2000)
    parser.add_argument('--toplayer_lr',type=float, default=0.1)
    parser.add_argument('--bert_lr',type=float, default=2e-3)
    parser.add_argument('--bert_warmup_steps', type=int, default=30000)
    parser.add_argument('--toplayer_warmup_steps', type=int, default=15000)
    parser.add_argument('--ckpt_path', type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_config()
    device = args.gpu_id

    directory = args.ckpt_path
    import os
    if os.path.exists(directory):
        pass
    else: # recursively construct directory
        os.makedirs(directory, exist_ok=True)
    summary_dir = './decoded_result/'
    model_dir = './reference_result/'

    # Data Preparation
    bert_model, bert_vocab = init_bert_model(args.model_name)
    train_path, dev_path = args.train_data, args.dev_data
    data = Data(train_path, dev_path, bert_vocab, args.seq_max_len, args.tgt_vocab_f, args.tgt_vocab_size)
    vocab_size = data.tgt_vocab_size
    print ('Vocabulary Size is %d' % vocab_size)
    print ('data is ready')

    padding_idx = data.tgt_padding_idx
    # myModel construction
    print ('Initializing model...')
    embed_dim = bert_model.config.hidden_size
    model = NAG_BERT(bert_model, vocab_size, embed_dim, args.crf_low_rank, args.crf_beam_size, 
        args.dropout, data.src_padding_idx, data.tgt_padding_idx)
    model = model.cuda(args.gpu_id)
    print ('Finished model initialization')

    toplayer_optimizer = torch.optim.Adam(model.toplayer.parameters(), args.toplayer_lr)
    bert_optimizer = torch.optim.Adam(model.bert_model.parameters(), args.bert_lr)

    NLL = torch.nn.NLLLoss(ignore_index = padding_idx, reduction = 'none')

    #--- training part ---#
    batch_size = args.batch_size
    num_epochs = args.number_epoch
    training_data_num, dev_data_num = data.train_num, data.dev_num
    train_step_num = int(training_data_num / batch_size) + 1
    dev_step_num = int(dev_data_num / batch_size) + 1
    max_dev_bleu, min_dev_ppl = 0.0, 1e10

    max_dev_combine_rogue = -1.0

    batch_acm = 0
    curr_gradient_acc_step = 0
    for epoch in range(num_epochs):
        loss_accumulated = 0.
        model.train()
        print ('------------------------------------------------------------------------------------')
        if epoch % 5 == 0:
            print ('%d epochs have run' % epoch)
        else:
            pass
        batches_processed = 0
        train_acc_accumulated = 0.0

        for train_step in range(train_step_num):
            batch_acm += 1

            # optimizer warmups
            toplayer_lr_ratio = compute_lr(batch_acm, args.toplayer_warmup_steps)
            update_lr(toplayer_optimizer, args.toplayer_lr*toplayer_lr_ratio)
            bert_lr_ratio = compute_lr(batch_acm, args.bert_warmup_steps)
            update_lr(bert_optimizer, args.bert_lr*bert_lr_ratio)

            is_training = True
            #print (train_step)
            batches_processed += 1
            toplayer_optimizer.zero_grad()
            bert_optimizer.zero_grad()

            train_batch_src_inp, train_batch_tgt, _ = data.get_next_batch(batch_size, mode = 'train')
            train_batch_src_inp = torch.LongTensor(train_batch_src_inp).cuda(device)
            train_batch_tgt = torch.LongTensor(train_batch_tgt).cuda(device)
            bsz, tgt_len = train_batch_tgt.size()

            train_log_prob_matrix, train_batch_crf_loss = model(train_batch_src_inp, train_batch_tgt, is_training = True)
            # train_log_prob_matrix : tgt_len x bsz x vocab_size
            train_log_prob_matrix_unbind = torch.unbind(train_log_prob_matrix, dim = 0)
            assert len(train_log_prob_matrix_unbind) == tgt_len
            train_truth_tgt_unbind = torch.unbind(train_batch_tgt, dim = 1)
            assert len(train_truth_tgt_unbind) == tgt_len

            nll_loss_list = []

            for index in range(tgt_len):
                curr_step_log_prob = train_log_prob_matrix_unbind[index]
                curr_step_tgt = train_truth_tgt_unbind[index].view(bsz)
                one_nll_loss = NLL(curr_step_log_prob, curr_step_tgt)
                assert one_nll_loss.size() == torch.Size([bsz])
                nll_loss_list.append(one_nll_loss)

            nll_loss_matrix = torch.stack(nll_loss_list, dim = 1)
            assert nll_loss_matrix.size() == torch.Size([bsz, tgt_len])
            tgt_padding_matrix = ~train_batch_tgt.eq(data.tgt_padding_idx)
            tgt_padding_matrix = tgt_padding_matrix.type(nll_loss_matrix.type())
            assert tgt_padding_matrix.size() == nll_loss_matrix.size()
            nll_loss_matrix = nll_loss_matrix * tgt_padding_matrix

            train_nll_loss = nll_loss_matrix.sum(-1)
            assert train_batch_crf_loss.size() == train_nll_loss.size()
            one_step_train_loss = train_batch_crf_loss + args.nll_loss_weight * train_nll_loss
            train_loss = torch.sum(one_step_train_loss) / torch.sum(tgt_padding_matrix)

            one_loss_accumulated = torch.sum(nll_loss_matrix) / torch.sum(tgt_padding_matrix)
            loss_accumulated += one_loss_accumulated.item()

            ###
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            toplayer_optimizer.step()
            bert_optimizer.step()
            
            if batches_processed % args.print_every == 0:
                curr_train_loss = loss_accumulated / args.print_every
                curr_train_ppl = np.exp(curr_train_loss)
                print ("At epoch %d, batch %d, current training loss is %.5f, \
                    corresponding training ppl is %.5f" % (epoch, train_step, curr_train_loss, 
                        curr_train_ppl))
                loss_accumulated = 0.

            if batches_processed % args.eval_every == 0:
                reference = []
                hypothesis = []

                dev_loss_acm = 0.0
                with torch.no_grad():
                    for dev_step in range(dev_step_num):
                        is_training = False
                        dev_batch_src_inp, dev_batch_tgt, dev_batch_truth \
                        = data.get_next_batch(batch_size, mode = 'dev')

                        dev_batch_src_inp = torch.LongTensor(dev_batch_src_inp).cuda(device)
                        dev_batch_tgt = torch.LongTensor(dev_batch_tgt).cuda(device)
                        dev_batch_source_text = get_article_ref_text(dev_batch_src_inp, data)
                        reference += dev_batch_truth 

                        _, one_tgt_len = dev_batch_tgt.size()

                        dev_batch_greedy_result = model.decoding(dev_batch_src_inp)

                        dev_batch_hypothesis = map_text(dev_batch_greedy_result, data)
                        hypothesis += dev_batch_hypothesis

                        # ------------------------------------------------------------------
                        # evaluate dev ppl
                        #dev_batch_tgt = torch.LongTensor(dev_batch_tgt).cuda(device)
                        bsz, tgt_len = dev_batch_tgt.size()

                        dev_log_prob_matrix, _ = model(dev_batch_src_inp, dev_batch_tgt, is_training = False)

                        dev_log_prob_matrix_unbind = torch.unbind(dev_log_prob_matrix, dim = 0)
                        assert len(dev_log_prob_matrix_unbind) == tgt_len
                        dev_truth_tgt_unbind = torch.unbind(dev_batch_tgt, dim = 1)
                        assert len(dev_truth_tgt_unbind) == tgt_len

                        nll_loss_list = []
                        for index in range(tgt_len):
                            curr_step_log_prob = dev_log_prob_matrix_unbind[index]
                            curr_step_tgt = dev_truth_tgt_unbind[index].view(bsz)
                            one_nll_loss = NLL(curr_step_log_prob, curr_step_tgt)
                            assert one_nll_loss.size() == torch.Size([bsz])
                            nll_loss_list.append(one_nll_loss)

                        nll_loss_matrix = torch.stack(nll_loss_list, dim = 1)
                        assert nll_loss_matrix.size() == torch.Size([bsz, tgt_len])
                        tgt_padding_matrix = ~dev_batch_tgt.eq(data.tgt_padding_idx)
                        tgt_padding_matrix = tgt_padding_matrix.type(nll_loss_matrix.type())
                        assert tgt_padding_matrix.size() == nll_loss_matrix.size()
                        nll_loss_matrix = nll_loss_matrix * tgt_padding_matrix

                        dev_loss = torch.sum(nll_loss_matrix) / torch.sum(tgt_padding_matrix)

                        dev_loss_acm += dev_loss.item()
                        # ------------------------------------------------------------------

                    # --------------------------------------------------------------------------------------------
                    # evaluate rogue score
                    # write generated result
                    create_directory(summary_dir)
                    create_directory(model_dir)
                    write_results(summary_dir, hypothesis, mode = r'decoded')
                    # write reference result
                    write_results(model_dir, reference, mode = r'reference')
                    # compute score
                    one_rogue_1_score, one_rogue_2_score, one_rogue_l_score = \
                    get_rouge_scores(summary_dir, model_dir)
                    one_combine_rogue = one_rogue_1_score + one_rogue_2_score + one_rogue_l_score

                    rm_tmp_files()

                    curr_dev_loss = dev_loss_acm / dev_step_num
                    curr_dev_ppl = np.exp(curr_dev_loss)

                    #if one_combine_rogue > max_dev_combine_rogue or curr_dev_ppl < min_dev_ppl:
                    if one_combine_rogue > max_dev_combine_rogue:
                        if one_combine_rogue > max_dev_combine_rogue:
                            max_dev_combine_rogue = one_combine_rogue

                        print ('Save Model.')

                        torch.save({'args':args, 'model':model.state_dict(), 
                                }, directory + '/best_ckpt')

                        fileData = {}
                        for fname in os.listdir(directory):
                            fileData[fname] = os.stat(directory + '/' + fname).st_mtime
                        sortedFiles = sorted(fileData.items(), key=itemgetter(1))
                        if len(sortedFiles) < 1:
                            pass
                        else:
                            delete = len(sortedFiles) - 1
                            for x in range(0, delete):
                                os.remove(directory + '/' + sortedFiles[x][0])

                    if curr_dev_ppl < min_dev_ppl:
                        min_dev_ppl = curr_dev_ppl
                    # --------------------------------------------------------------------------------------------

                    print ('##########################################################################')
                    print ("At epoch %d, batch %d, current dev rogue-1 is %.5f, rogue-2 is %.5f, rogue-l is %.5f, combine rogue is %.5f, curr dev ppl is %f" \
                        % (epoch, train_step, one_rogue_1_score, one_rogue_2_score, one_rogue_l_score, one_combine_rogue, curr_dev_ppl))
                    print ("Corresponding maximum dev combine rogue is %.5f, min dev ppl is %5f" % (max_dev_combine_rogue, min_dev_ppl))
                    print ('##########################################################################')
                model.train()
