#!/usr/bin/env bash

if [ ! -d env ]; then
    python3 -m venv env
fi

source env/bin/activate

pip install torch==1.7.1
pip install jupyter
pip install subword-nmt
pip install sacremoses
pip install googletrans==3.1.0a0
pip install pandas
pip install sacrebleu
pip install matplotlib

bash download-data.sh
