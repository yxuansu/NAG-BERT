import random
import numpy as np
import progressbar
PAD, UNK, SEP, CLS, SOS, EOS = '[PAD]', '[UNK]', '[SEP]', '[CLS]', '[unused0]', '[unused1]'

class Vocab:
    def __init__(self, vocab_f, vocab_size):
        special_tokens = [PAD, UNK, EOS]
        max_vocab_size = vocab_size + 3
        self.token_idx_dict = {}
        self.idx_token_dict = {}
        self.vocab_list = []
        index = 0
        for token in special_tokens:
            self.token_idx_dict[token] = index
            self.idx_token_dict[index] = token
            self.vocab_list.append(token)
            index += 1

        with open(vocab_f, 'r', encoding = 'utf8') as i:
            lines = i.readlines()
            for l in lines:
                if index >= max_vocab_size:
                    continue
                else:
                    pass
                token = l.strip('\n').split()[0]
                self.token_idx_dict[token] = index
                self.idx_token_dict[index] = token
                self.vocab_list.append(token)

                index += 1

        self.padding_idx = self.token_idx_dict[PAD]
        self.unk_idx = self.token_idx_dict[UNK]
        self.eos_idx = self.token_idx_dict[EOS]
        self.vocab_size = len(self.token_idx_dict)
        assert len(self.vocab_list) == self.vocab_size

    def tokentoidx(self, in_list):
        out_list = []
        for token in in_list:
            try:
                out_list.append(self.token_idx_dict[token])
                assert self.token_idx_dict[token] < self.vocab_size
            except KeyError:
                continue
                #out_list.append(self.unk_idx)
                #assert self.unk_idx < self.vocab_size
        return out_list

    def idxtotoken(self, in_idx_list):
        out_list = []
        for idx in in_idx_list:
            out_list.append(self.idx_token_dict[idx])
        return out_list



class Data:
    def __init__(self, train_path, dev_path, bert_vocab, seq_max_len, tgt_vocab_f, tgt_vocab_size):
        '''
            sen1_list is post list
            sen2_list is response list
        '''
        self.train_path = train_path
        self.dev_path = dev_path
        self.vocab = bert_vocab
        self.vocab_size = len(self.vocab.vocab)

        self.tgt_vocab = Vocab(tgt_vocab_f, tgt_vocab_size)
        self.tgt_vocab_size = self.tgt_vocab.vocab_size
        self.tgt_padding_idx = self.tgt_vocab.padding_idx

        self.src_padding_idx, self.unk_idx, self.sep_idx, self.cls_idx, self.sos_idx, self.eos_idx = \
        self.vocab.convert_tokens_to_ids([PAD, UNK, SEP, CLS, SOS, EOS])
        self.seq_max_len = seq_max_len

        self.train_article_list, self.train_title_list = self.load_data(train_path, mode='train')
        self.dev_article_list, self.dev_title_list = self.load_data(dev_path, mode='dev')

        self.train_num, self.dev_num = len(self.train_article_list), len(self.dev_article_list)
        print ('train number is %d, dev number is %d' % (self.train_num, self.dev_num))
        print ('tgt vocab size is %d' % self.tgt_vocab_size)

        self.train_idx_list, self.dev_idx_list = [i for i in range(self.train_num)], [j for j in range(self.dev_num)]
        self.shuffle_train_idx()

        self.train_current_idx = 0
        self.dev_current_idx = 0

    def vocab_map_text(self, text):
        text_list = text.strip().split()
        text_idx_list = self.vocab.convert_tokens_to_ids(text_list)
        clean_text_list = self.vocab.convert_ids_to_tokens(text_idx_list)
        res_text_list = []
        for token in clean_text_list:
            if token == UNK:
                continue
            else:
                res_text_list.append(token)
        return res_text_list

    def load_data(self, path, mode):
        article_list, title_list = [], []
        with open(path, 'r', encoding = 'utf8') as i:
            if mode == 'train':
                lines = i.readlines()
                print ('Start loading training data...')
            elif mode == 'dev':
                print ('Start loading validation data...')
                lines = i.readlines()[:2000]
            else:
                raise Exception('Wrong Data Mode!!!')

            data_num = len(lines)
            p = progressbar.ProgressBar(data_num)
            p.start()
            idx = 0
            for l in lines:
                p.update(idx)
                idx += 1
                content_list = l.strip('\n').split('\t')
                assert len(content_list) == 2
                one_article, one_title = content_list

                one_article_list = self.vocab.tokenize(one_article.strip())
                one_title_list = one_title.strip().split()

                tgt_valid_len = len(self.tgt_vocab.tokentoidx(one_title_list))
                if tgt_valid_len == 0:
                    continue

                if len(one_article_list) <= tgt_valid_len:
                    continue

                if len(one_article_list) >= self.seq_max_len or tgt_valid_len >= self.seq_max_len:
                    continue
                else:
                    pass
                    
                if len(one_article_list) == 0 or len(one_title_list) == 0:
                    continue
                else:
                    pass

                article_list.append(one_article_list)
                title_list.append(one_title_list)
            p.finish()
        return article_list, title_list

    def shuffle_train_idx(self):
        random.shuffle(self.train_idx_list)

    def shuffle_dev_idx(self):
        random.shuffle(self.dev_idx_list)

    def pad_data(self, batch_in_token_list, max_len, padding_idx):
        len_list = [len(item) for item in batch_in_token_list]
        batch_out_token_list = []
        for i in range(len(batch_in_token_list)):
            one_len = len_list[i]
            len_diff = max_len - one_len
            one_out = batch_in_token_list[i] + [padding_idx for _ in range(len_diff)]
            #one_out = batch_in_token_list[i] + [PAD for _ in range(len_diff)]
            batch_out_token_list.append(one_out)
        return batch_out_token_list

    def get_next_batch(self, batch_size, mode):
        batch_inp_list, batch_tgt_list, batch_truth_list = [], [], []

        if mode == 'train':
            if self.train_current_idx + batch_size < self.train_num - 1:
                for i in range(batch_size):
                    curr_idx = self.train_current_idx + i
                    one_inp_token_list = self.train_article_list[self.train_idx_list[curr_idx]]
                    one_inp_token_list = [CLS] + one_inp_token_list + [SEP]
                    batch_inp_list.append(self.vocab.convert_tokens_to_ids(one_inp_token_list))

                    one_tgt_token_list = self.train_title_list[self.train_idx_list[curr_idx]]
                    batch_truth_list.append(' '.join(one_tgt_token_list).strip())

                    one_tgt_token_list = one_tgt_token_list + [EOS] + [EOS] # two EOS for CRF learning
                    batch_tgt_list.append(self.tgt_vocab.tokentoidx(one_tgt_token_list))
                    assert len(one_inp_token_list) >= len(one_tgt_token_list)
                self.train_current_idx += batch_size
            else:
                len_diff = self.train_num - self.train_current_idx
                for i in range(batch_size):
                    curr_idx = self.train_current_idx + i
                    if curr_idx > self.train_current_idx - 1:
                        self.shuffle_train_idx()
                        curr_idx = 0
                        one_inp_token_list = self.train_article_list[self.train_idx_list[curr_idx]]
                        one_inp_token_list = [CLS] + one_inp_token_list + [SEP]
                        batch_inp_list.append(self.vocab.convert_tokens_to_ids(one_inp_token_list))

                        one_tgt_token_list = self.train_title_list[self.train_idx_list[curr_idx]]
                        batch_truth_list.append(' '.join(one_tgt_token_list).strip())

                        one_tgt_token_list = one_tgt_token_list + [EOS] + [EOS] # two EOS for CRF learning
                        batch_tgt_list.append(self.tgt_vocab.tokentoidx(one_tgt_token_list))
                        assert len(one_inp_token_list) >= len(one_tgt_token_list)
                    else:
                        one_inp_token_list = self.train_article_list[self.train_idx_list[curr_idx]]
                        one_inp_token_list = [CLS] + one_inp_token_list + [SEP]
                        batch_inp_list.append(self.vocab.convert_tokens_to_ids(one_inp_token_list))

                        one_tgt_token_list = self.train_title_list[self.train_idx_list[curr_idx]]
                        batch_truth_list.append(' '.join(one_tgt_token_list).strip())

                        one_tgt_token_list = one_tgt_token_list + [EOS] + [EOS] # two EOS for CRF learning
                        batch_tgt_list.append(self.tgt_vocab.tokentoidx(one_tgt_token_list))
                        assert len(one_inp_token_list) >= len(one_tgt_token_list)
                self.train_current_idx = 0

        elif mode == 'dev':
            if self.dev_current_idx + batch_size < self.dev_num - 1:
                for i in range(batch_size):
                    curr_idx = self.dev_current_idx + i

                    one_inp_token_list = self.dev_article_list[curr_idx]
                    one_inp_token_list = [CLS] + one_inp_token_list + [SEP]
                    batch_inp_list.append(self.vocab.convert_tokens_to_ids(one_inp_token_list))

                    one_tgt_token_list = self.dev_title_list[curr_idx]
                    batch_truth_list.append(' '.join(one_tgt_token_list).strip())

                    one_tgt_token_list = one_tgt_token_list + [EOS] + [EOS]
                    batch_tgt_list.append(self.tgt_vocab.tokentoidx(one_tgt_token_list))
                    assert len(one_inp_token_list) >= len(one_tgt_token_list)
                self.dev_current_idx += batch_size
            else:
                for i in range(batch_size):
                    curr_idx = self.dev_current_idx + i
                    if curr_idx > self.dev_num - 1: # 对dev_current_idx重新赋值
                        curr_idx = 0
                        self.dev_current_idx = 0
                    else:
                        pass

                    one_inp_token_list = self.dev_article_list[curr_idx]
                    one_inp_token_list = [CLS] + one_inp_token_list + [SEP]
                    batch_inp_list.append(self.vocab.convert_tokens_to_ids(one_inp_token_list))

                    one_tgt_token_list = self.dev_title_list[curr_idx]
                    batch_truth_list.append(' '.join(one_tgt_token_list).strip())

                    one_tgt_token_list = one_tgt_token_list + [EOS] + [EOS]
                    batch_tgt_list.append(self.tgt_vocab.tokentoidx(one_tgt_token_list))
                    assert len(one_inp_token_list) >= len(one_tgt_token_list)
                self.dev_current_idx = 0
                #self.shuffle_dev_idx()
        else:
            raise Exception('Wrong batch mode!!!')

        max_len = max([len(item) for item in batch_inp_list])
        batch_src_inp_pad = self.pad_data(batch_inp_list, max_len, self.src_padding_idx)
        batch_tgt_inp_pad = self.pad_data(batch_tgt_list, max_len, self.tgt_padding_idx)
        return batch_src_inp_pad, batch_tgt_inp_pad, batch_truth_list

