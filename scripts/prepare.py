#!/usr/bin/env python3

import os
import argparse
import random
import requests
import io
from zipfile import ZipFile
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument('langs', nargs='+')
parser.add_argument('--data-dir', default='data')
args = parser.parse_args()

assert not any(lang == 'en' for lang in args.langs)

random.seed(1234)

test_size = 2000
valid_size = 2000

args.langs = sorted(set(args.langs))

def preprocess(line, lang):
    return ' '.join(line.split())


def read_corpus(src_lang, tgt_lang='en'):
    assert src_lang != tgt_lang
    pair = '-'.join(sorted([src_lang, tgt_lang]))
    url = f'https://object.pouta.csc.fi/OPUS-Tatoeba/v2023-04-12/moses/{pair}.txt.zip'
    src_filename = f'Tatoeba.{pair}.{src_lang}'
    tgt_filename = f'Tatoeba.{pair}.{tgt_lang}'
    print(f'Downloading {pair} data from {url}')
    zip_file = ZipFile(io.BytesIO(requests.get(url).content))
    corpus = OrderedDict()
    print(f'Pre-processing {pair} data')
    with zip_file.open(src_filename) as src_file, zip_file.open(tgt_filename) as tgt_file:
        for src, tgt in zip(src_file, tgt_file):
            src = preprocess(src.decode(), src_lang)
            tgt = preprocess(tgt.decode(), tgt_lang)
            if not src or not tgt:
                continue
            corpus[tgt] = src
    return corpus

corpora = {lang: read_corpus(lang) for lang in args.langs}

print(f"Creating splits in '{args.data_dir}'")
os.makedirs(args.data_dir, exist_ok=True)

en_lines = list(corpora.values())
en_lines = list(OrderedDict((en, None) for en in en_lines[0] if all(en in lines for lines in en_lines[1:])))  # intersection of all corpora

random.shuffle(en_lines)

splits = {'test': 2000, 'valid': 2000, 'train': None}

for split_name, split_size in splits.items():

    en_lines_ = en_lines[:split_size]
    en_lines = en_lines[split_size:]

    for src_lang in args.langs + ['en']:
        for tgt_lang in args.langs + ['en']:
            if src_lang == tgt_lang:
                continue
            
            if src_lang == 'en':
                src_lines = en_lines_
            else:
                src_lines = [corpora[src_lang][en] for en in en_lines_]

            if tgt_lang == 'en':
                tgt_lines = en_lines_
            else:
                tgt_lines = [corpora[tgt_lang][en] for en in en_lines_]

            src_filename = os.path.join(args.data_dir, f'{split_name}.{src_lang}-{tgt_lang}.{src_lang}')
            tgt_filename = os.path.join(args.data_dir, f'{split_name}.{src_lang}-{tgt_lang}.{tgt_lang}')
            with open(src_filename, 'w') as src_file, open(tgt_filename, 'w') as tgt_file:
                src_file.writelines(line + '\n' for line in src_lines)
                tgt_file.writelines(line + '\n' for line in tgt_lines)

print('Finished')
