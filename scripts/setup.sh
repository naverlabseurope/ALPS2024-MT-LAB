#!/usr/bin/env bash

if [ ! -d env ]; then
    python3 -m venv env
fi

source env/bin/activate

pip install torch
pip install jupyter
# pip install sentencepiece
pip install subword_nmt
pip install sacremoses
pip install googletrans==3.1.0a0
pip install pandas
pip install sacrebleu
pip install matplotlib
pip install requests
pip install bertviz

bash scripts/download-data.sh
