import os
import sys
import json
import random
import logging
import tensorflow as tf
import sentencepiece as spm

from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser


# configure logging
log = logging.getLogger('tensorflow')
log.setLevel(logging.INFO)


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
    parser.add_argument('--train-dataset', type=Path)
    parser.add_argument('--validation-dataset', type=Path)
    parser.add_argument('--vocab-file')
    parser.add_argument('--vocab-size', type=int, default=32000)
    parser.add_argument('--subsample', type=int, default=12800000)
    parser.add_argument('--model-prefix', default='tokenizer')
    parser.add_argument('--processed-output-directory', type=Path, dest='output')
    parser.add_argument('--do-lower-case', action='store_true')

    args = parser.parse_args()

    with args.train_dataset.open(encoding="utf-8") as fp:
        with (args.output / 'train.txt').open('w', encoding='utf-8') as fo:
            for l in tqdm(fp):
                if args.do_lower_case:
                    l = str(l).lower()
                fo.write(l+"\n")

    with args.validation_dataset.open(encoding="utf-8") as fp:
        with (args.output / 'validation.txt').open('w', encoding='utf-8') as fo:
            for l in tqdm(fp):
                if args.do_lower_case:
                    l = str(l).lower()
                fo.write(l+"\n")

    SPM_COMMAND = ('--input={infile} --model_prefix={prefix} '
                   '--vocab_size={vocab_size} --input_sentence_size={subsample_size} '
                   '--shuffle_input_sentence=true '
                   '--bos_id=-1 --eos_id=-1').format(
        infile=(args.output / 'train.txt'), prefix=args.model_prefix,
        vocab_size=args.vocab_size-256, subsample_size=args.subsample
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
            fp.write(token + '\n')
