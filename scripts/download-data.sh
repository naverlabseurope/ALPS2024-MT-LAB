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

# # Train a multilingual (de, fr, en) BPE model
# if [ ! -f data/spm.de-en-fr.model ]; then
#     cat data/train.en-fr.{en,fr} data/train.de-en.de | scripts/train_spm.py data/spm.de-en-fr -s 8000
# fi

# Download data for other languages (e.g., italian). Replace with the language of your choice.
# Warning: avoid re-downloading de and fr data paired with other languages as this will result in different test splits
# if [ ! -f data/train.en-it.it ]; then
#     python3 scripts/prepare.py it --data-dir data
# fi

# TODO: upload new pre-trained models
# # Download pre-trained model from Dropbox
# if [ ! -f models/en-fr/pretrained-transformer.pt ]; then
#     wget -nc https://www.dropbox.com/s/14cxqgfaahagprl/models.zip
#     unzip models.zip
# fi
