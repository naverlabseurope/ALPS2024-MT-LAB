#!/usr/bin/env bash

if [ ! -d env ]; then
    python3 -m venv env
fi

source env/bin/activate

pip install jupyter

source env/bin/activate
jupyter notebook --ip 0.0.0.0 --port 8888 NMT.ipynb
