# NAG-BERT
[Non-Autoregressive Text Generation with Pre-trained Language Models](https://arxiv.org/abs/2102.08220)

Authors: Yixuan Su, Deng Cai, Yan Wang, David Vandyke, Simon Baker, Piji Li, and Nigel Collier

## Introduction:
In this repository, we provide the related resources to our EACL 2021 paper. We provide training and inference code for text summarization task.

## 1. Enviornment Installtion:
```yaml
pip install -r requirements.txt
```
To install pyrouge, please refer to this [link](https://sagor-sarker.medium.com/how-to-install-rouge-pyrouge-in-ubuntu-16-04-7f0ec1cda81b)

## 2. Download Gigawords Data [here](https://drive.google.com/file/d/1Jx9yfx45UJmFsO6y9lBlkGshPD3tF8Xy/view?usp=sharing):
```yaml
unzip data.zip and replace it with the empty ./data folder.
```

## 3. Training
```yaml
chmod +x ./train.sh
./train.sh
```

## 4. Inference

```yaml
chmod +x ./inference.sh
./inference.sh

The $\alpha$ in the ratio-first decoding can be controlled by changing the value of --length_ratio
```

## 5. Citation
If you find our paper and resources useful, please kindly cite our paper:
```yaml
@inproceedings{su-etal-2021-non,
    title = "Non-Autoregressive Text Generation with Pre-trained Language Models",
    author = "Su, Yixuan  and
      Cai, Deng  and
      Wang, Yan  and
      Vandyke, David  and
      Baker, Simon  and
      Li, Piji  and
      Collier, Nigel",
    booktitle = "Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume",
    month = apr,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.eacl-main.18",
    pages = "234--243"
}
```

## Acknowledgements
The authors would like to thank [Huggingface](https://huggingface.co/) and [Fairseq](https://github.com/pytorch/fairseq) for making their awesome codes publicly available. Some of our codes are borrowed from these libraries.

