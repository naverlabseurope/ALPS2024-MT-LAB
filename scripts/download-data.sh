#!/usr/bin/env bash

if [ -d env ]; then
    source env/bin/activate
fi

# Download and pre-process multi-parallel (de, fr, en) data from OPUS
if [ ! -f data/train.en-fr.fr ]; then
    python3 scripts/prepare.py de fr --data-dir data
fi

# Train a multilingual (de, fr, en) BPE model
if [ ! -f data/spm.de-en-fr.model ]; then
    cat data/train.en-fr.{en,fr} data/train.de-en.de > data/train.concat
    python3 -c "import sentencepiece as spm; spm.SentencePieceTrainer.Train(input='data/train.concat', model_type='bpe', model_prefix='data/spm.de-en-fr', vocab_size=8000)"
    rm data/train.concat
fi

# Download data for other languages (e.g., Italian). Replace with the language of your choice.
# Warning: avoid re-downloading de and fr data paired with other languages as this will result in different test splits
# if [ ! -f data/train.en-it.it ]; then
#     python3 scripts/prepare.py it --data-dir data
# fi

# Download pre-trained models from Dropbox
if [ ! -f pretrained_models/en-fr/transformer.pt ]; then
    wget -nc https://www.dropbox.com/s/ckdxt6h8hj4lygw/pretrained_models.zip
    unzip pretrained_models.zip
fi
