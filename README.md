# ERNIE_pytorch

[ERNIE](https://github.com/PaddlePaddle/LARK/tree/develop/ERNIE) is a Language Model from Baidu. ERNIE is implemented by PaddlePaddle, with script in this git repo, you can convert ERNIE to pytorch version (Based on [huggingface's implementation](https://github.com/huggingface/pytorch-pretrained-BERT)).

Firstly you need to download ERNIE model and convert it to numpy array using script in [this issue](https://github.com/PaddlePaddle/LARK/issues/37). Then run:

`python convert.py`

Be aware of that, ERNIE uses different vocab.txt compared with BERT.

We provide a simple interface to see ERNIE's effect, you can run:

`python mask.py`