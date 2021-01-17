#!/usr/bin/env python

import sacremoses
import subword_nmt
import random
from collections import OrderedDict

random.seed(1234)

tokenizers = {lang: sacremoses.MosesTokenizer(lang=lang) for lang in ('en', 'fr', 'de')}

def preprocess(line, lang):
    tokenizer = tokenizers[lang]
    return tokenizer.tokenize(line.strip(), escape=False, return_str=True).lower()

def read_corpus(filename, lang):
    corpus = []
    with open(filename) as f:
        for line in f:
            src, tgt, *_ = line.split('\t')
            src = preprocess(src, 'en')
            tgt = preprocess(tgt, lang)
            corpus.append((src, tgt))
    random.shuffle(corpus)
    return corpus

en_fr_corpus = read_corpus('data/fra.txt', 'fr')
en_de_corpus = read_corpus('data/deu.txt', 'de')

en_fr_index = {en: fr for en, fr in en_fr_corpus}
en_de_index = {en: de for en, de in en_de_corpus}

en_lines = OrderedDict((en, None) for en in list(en_fr_index) + list(en_de_index) if en in en_fr_index and en in en_de_index)
# using OrderedDict instead of set for reproducibility
en_lines = list(en_lines)
random.shuffle(en_lines)
en_valid = en_lines[:2000]
en_test = en_lines[2000:4000]
en_train = en_lines[4000:]

to_exclude = set(en_valid + en_test)

en_fr_train = [(en, fr) for en, fr in en_fr_corpus if en not in to_exclude]
en_de_train = [(en, de) for en, de in en_de_corpus if en not in to_exclude]
de_fr_train = [(en_de_index[en], en_fr_index[en]) for en in en_train]

en_fr_valid = [(en, en_fr_index[en]) for en in en_valid]
en_de_valid = [(en, en_de_index[en]) for en in en_valid]
de_fr_valid = [(en_de_index[en], en_fr_index[en]) for en in en_valid]

en_fr_test = [(en, en_fr_index[en]) for en in en_test]
en_de_test = [(en, en_de_index[en]) for en in en_test]
de_fr_test = [(en_de_index[en], en_fr_index[en]) for en in en_test]

def save_corpus(corpus, src_filename, tgt_filename):
    with open(src_filename, 'w') as src_file, open(tgt_filename, 'w') as tgt_file:
        for src, tgt in corpus:
            src_file.write(src + '\n')
            tgt_file.write(tgt + '\n')

save_corpus(en_fr_train, 'data/train.en-fr.en', 'data/train.en-fr.fr')
save_corpus(en_de_train, 'data/train.de-en.en', 'data/train.de-en.de')
save_corpus(de_fr_train, 'data/train.de-fr.de', 'data/train.de-fr.fr')

save_corpus(en_fr_valid, 'data/valid.en-fr.en', 'data/valid.en-fr.fr')
save_corpus(en_de_valid, 'data/valid.de-en.en', 'data/valid.de-en.de')
save_corpus(de_fr_valid, 'data/valid.de-fr.de', 'data/valid.de-fr.fr')

save_corpus(en_fr_test, 'data/test.en-fr.en', 'data/test.en-fr.fr')
save_corpus(en_de_test, 'data/test.de-en.en', 'data/test.de-en.de')
save_corpus(de_fr_test, 'data/test.de-fr.de', 'data/test.de-fr.fr')
