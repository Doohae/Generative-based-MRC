from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

import random
import numpy as np
import torch


def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    

def get_model(arg):

    model_name = arg.model_name

    # T5는 seq2seq 모델이므로 model을 불러올 때 AutoModelForSeq2SeqLM을 사용해야 함
    config = AutoConfig.from_pretrained(
        model_name,
        cache_dir=None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=None,
        use_fast=True,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        config=config,
        cache_dir=None,
    )

    return model, tokenizer