#!/usr/bin/env python

import sacremoses
import subword_nmt
import random
import argparse
import os
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument('path')   # data/ita.txt for example (after downloading and extracting http://www.manythings.org/anki/ita-eng.zip)

args = parser.parse_args()

source_lang = 'en'
target_lang = os.path.basename(args.path).split('.')[0][:2]  # data/ita.txt -> it
dirname = os.path.dirname(args.path)  # data/ita.txt -> data

random.seed(1234)

tokenizers = {lang: sacremoses.MosesTokenizer(lang=lang) for lang in (source_lang, target_lang)}

def preprocess(line, lang):
    tokenizer = tokenizers[lang]
    return tokenizer.tokenize(line.strip(), escape=False, return_str=True).lower()

corpus = []
with open(args.path) as f:
    for line in f:
        src, tgt, *_ = line.split('\t')
        src = preprocess(src, source_lang)
        tgt = preprocess(tgt, target_lang)
        corpus.append((src, tgt))
random.shuffle(corpus)

source_lang, target_lang = sorted([source_lang, target_lang])

source_suffix = '.{}-{}.{}'.format(*sorted([source_lang, target_lang]), source_lang)
target_suffix = '.{}-{}.{}'.format(*sorted([source_lang, target_lang]), target_lang)

def save_corpus(corpus, src_filename, tgt_filename):
    with open(src_filename, 'w') as src_file, open(tgt_filename, 'w') as tgt_file:
        for src, tgt in corpus:
            src_file.write(src + '\n')
            tgt_file.write(tgt + '\n')

valid_corpus = corpus[:2000]
test_corpus = corpus[2000:4000]
train_corpus = corpus[4000:]

save_corpus(
    valid_corpus,
    os.path.join(dirname, 'valid' + source_suffix),
    os.path.join(dirname, 'valid' + target_suffix)
)

save_corpus(
    test_corpus,
    os.path.join(dirname, 'test' + source_suffix),
    os.path.join(dirname, 'test' + target_suffix)
)

save_corpus(
    train_corpus,
    os.path.join(dirname, 'train' + source_suffix),
    os.path.join(dirname, 'train' + target_suffix)
)
