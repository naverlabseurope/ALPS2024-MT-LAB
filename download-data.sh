#!/usr/bin/env bash

source env/bin/activate

mkdir -p data

# Download and pre-process multi-parallel (de, fr, en) data from http://www.manythings.org/anki/
if [ ! -f data/train.en-fr.fr ]; then
    pushd data
    wget -nc https://www.manythings.org/anki/fra-eng.zip
    wget -nc https://www.manythings.org/anki/deu-eng.zip
    unzip -o fra-eng.zip
    unzip -o deu-eng.zip
    popd
    python3 pyfiles/prepare-de-fr-en.py
fi

# Train a multilingual (de, fr, en) BPE model
if [ ! -f data/bpecodes.de-en-fr ]; then
    cat data/train.en-fr.{en,fr} data/train.de-en.de | subword-nmt learn-bpe -o data/bpecodes.de-en-fr -s 8000 -v
fi

# Download data for other languages (e.g., italian). Replace with the language of your choice.
if [ ! -f data/train.en-it.it ]; then
    pushd data
    wget -nc https://www.manythings.org/anki/ita-eng.zip
    unzip -o ita-eng.zip
    popd
    python3 pyfiles/prepare.py data/ita.txt
    cat data/train.en-it.{en,it} | subword-nmt learn-bpe -o data/bpecodes.en-it -s 8000 -v
    cat data/train.en-fr.{en,fr} data/train.de-en.de data/train.en-it.it | subword-nmt learn-bpe -o data/bpecodes.de-en-fr-it -s 8000 -v
fi

wget -nc https://www.dropbox.com/s/mxwamwnrs374bev/models.zip
unzip models.zip
