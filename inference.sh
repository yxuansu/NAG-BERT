CUDA_VISIBLE_DEVICES=1 python inference.py\
    --ckpt_path ./ckpt/\
    --tgt_vocab_f ./data/vocab.txt\
    --length_ratio 1.0\
    --test_path ./data/test.txt