CUDA_VISIBLE_DEVICES=1 python main.py\
    --train_data ./data/train.txt\
    --dev_data ./data/dev.txt\
    --tgt_vocab_f ./data/vocab.txt\
    --ckpt_path ./ckpt/