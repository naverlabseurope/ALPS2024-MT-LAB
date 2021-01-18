#!/usr/bin/env bash

source env/bin/activate
jupyter notebook --ip 0.0.0.0 --port 8888 NMT.ipynb
