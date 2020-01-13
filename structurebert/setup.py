import os
import sys
import json
import nltk
import random
import logging
import tensorflow as tf
import sentencepiece as spm
import tqdm as tqdm
import tempfile

from pathlib import Path
from argparse import ArgumentParser

sys.path.append("bert")

from bert import modeling, optimization, tokenization
from bert.run_pretraining import input_fn_builder, model_fn_builder

# configure logging
log = logging.getLogger('tensorflow')
log.setLevel(logging.INFO)


regex_tokenizer = nltk.RegexpTokenizer("\w+")

def normalize_text(text):
    # lowercase text
    text = str(text).lower()
    # remove non-UTF
    text = text.encode("utf-8", "ignore").decode()
    # remove punktuation symbols
    text = " ".join(regex_tokenizer.tokenize(text))

    return text


def read_sentencepiece_vocab(filepath):
    voc = []
    with open(filepath, encoding='utf-8') as fi:
        for line in fi:
            voc.append(line.split("\t")[0])
    # skip the first <unk> token and return
    return voc[1:]

def parse_sentencepiece_token(token):
    if token.startswith("▁"):
        return token[1:]
    else:
        return "##" + token


CONTROL_SYMBOLS = ["[PAD]","[UNK]","[CLS]","[SEP]","[MASK]"]

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-directory', type=Path)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--vocab-file')
    parser.add_argument('--vocab-size', type=int, default=32000)
    parser.add_argument('--subsample', type=int, default=12800000)
    parser.add_argument('--model-prefix', default='tokenizer')

    args = parser.parse_args()

    temp = tempfile.NamedTemporaryFile()
    log.info('Storing processed data in temp file:', temp.name)
    with args.dataset.open(encoding="utf-8") as fp:
        for l in tqdm(fi):
            temp.write(normalize_text(l)+"\n")
    temp.close()

    SPM_COMMAND = ('--input={infile} --model_prefix={prefix} '
                   '--vocab_size={vocab_size} --input_sentence_size={subsample_size} '
                   '--shuffle_input_sentence=true '
                   '--bos_id=-1 --eos_id=-1').format(
        infile=temp.name, prefix=args.model_prefix, 
        vocab_size=args.vocab_size-256, subsample_size=args.subsample_size
    )

    spm.SentencePieceTrainer.Train(SPM_COMMAND)

    vocab = read_sentencepiece_vocab("{}.vocab".format(args.model_prefix))
    log.info('Learned a vocabulary of size: {}'.format(len(vocab)))
    log.info('Sample tokens: {}'.format(random.sample(vocab, 10)))

    bert_vocab = [parse_sentencepiece_token(token) for token in vocab]
    bert_vocab = CONTROL_SYMBOLS + bert_vocab
    bert_vocab += ["[UNUSED_{}]".format(i) for i in range(args.vocab_size - len(bert_vocab))]

    log.info('Final vocabulary size for BERT: {}'.format(len(bert_vocab)))

    bert_base_config = {
        "attention_probs_dropout_prob": 0.1,
        "directionality": "bidi",
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "max_position_embeddings": 512,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "pooler_fc_size": 768,
        "pooler_num_attention_heads": 12,
        "pooler_num_fc_layers": 3,
        "pooler_size_per_head": 128,
        "pooler_type": "first_token_transform",
        "type_vocab_size": 2,
        "vocab_size": args.vocab_size
    }

    with (args.model_directory / 'bert_config.json').open('w') as fp:
        json.dump(bert_base_config, fp, indent=2)

    with (args.model_directory / args.vocab_file).open('w') as fp:
        for token in bert_vocab:
            fo.write(token + '\n')
