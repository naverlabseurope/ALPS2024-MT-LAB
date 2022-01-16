#!/usr/bin/env bash

source env/bin/activate

# Download and pre-process multi-parallel (de, fr, en) data from OPUS
if [ ! -f data/train.en-fr.fr ]; then
    python3 scripts/prepare.py de fr --data-dir data
fi

# Train a multilingual (de, fr, en) BPE model
if [ ! -f data/bpecodes.de-en-fr ]; then
    cat data/train.en-fr.{en,fr} data/train.de-en.de | subword-nmt learn-bpe -o data/bpecodes.de-en-fr -s 8000 -v
fi

# Download data for other languages (e.g., italian). Replace with the language of your choice.
# Warning: avoid re-downloading de and fr data paired with other languages as this will result in different test splits
# if [ ! -f data/train.en-it.it ]; then
#     python3 scripts/prepare.py it --data-dir data
# fi

# Download pre-trained models from Dropbox
if [ ! -f pretrained_models/en-fr/transformer.pt ]; then
    wget -nc https://www.dropbox.com/s/ckdxt6h8hj4lygw/pretrained_models.zip
    unzip pretrained_models.zip
fi
