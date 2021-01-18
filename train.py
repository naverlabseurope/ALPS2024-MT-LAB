#!/usr/bin/env python3

import os
import sys
sys.path.append('pyfiles')
import nmt_dataset
import nnet_models
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from functools import partial
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import copy
from subword_nmt.apply_bpe import BPE

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('checkpoint_path', help='Path to the output model checkpoint (e.g., models/my_model.pt)')
parser.add_argument('-s', '--source-lang', default='en', metavar='SRC', help='Source language (e.g., en)')
parser.add_argument('-t', '--target-lang', default='fr', metavar='TGT', help='Target language (e.g., fr)')
parser.add_argument('--source-dict', help='Path to the source dictionary. Default: dict.SRC.txt in the same directory as the checkpoint')
parser.add_argument('--target-dict', help='Path to the target dictionary. Default: dict.TGT.txt in the same directory as the checkpoint')
parser.add_argument('--bpecodes', help='Path to the BPE model. Default: DATA_DIR/bpecodes.de-en-fr')
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs. Default: 10')
parser.add_argument('--num-layers', type=int, default=1, help='Number of Transformer encoder and decoder layers. Default: 1')
parser.add_argument('--hidden-size', type=int, default=512, help='Hidden size and embedding size of the Transformer model. Default: 512')
parser.add_argument('--max-size', type=int, help='Maximum number of training examples. Default: all')
parser.add_argument('--data-dir', default='data', metavar='DATA_DIR', help='Directory containing the training data. Default: data')
parser.add_argument('--dropout', type=float, default=0, help='Dropout rate. Default: 0')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate. Default: 0.001')
parser.add_argument('--heads', type=int, default=4, help='Number of attention heads. Default: 4')
parser.add_argument('--batch-size', type=int, default=512, help='Maximum number of tokens in a batch. Default: 512')
parser.add_argument('--cpu', action='store_true', help='Train on CPU')
parser.add_argument('-v', '--verbose', action='store_true', help='Show training progress on a step-by-step basis')


args = parser.parse_args()

data_dir = args.data_dir
source_lang, target_lang = args.source_lang, args.target_lang
model_dir = os.path.dirname(args.checkpoint_path)
checkpoint_path = args.checkpoint_path
epochs = args.epochs
num_layers = args.num_layers
hidden_size = args.hidden_size
bpe_path = args.bpecodes or os.path.join(data_dir, 'bpecodes.de-en-fr')
max_size = args.max_size
minimum_count = 10
max_len = 30       # maximum 30 tokens per sentence (longer sequences will be truncated)
batch_size = args.batch_size
hidden_size = args.hidden_size
num_layers = args.num_layers
dropout = args.dropout
learning_rate = args.lr
attention_heads = args.heads
cpu = args.cpu
verbose = args.verbose

def reset_seed(seed=1234):
    np.random.seed(seed)
    torch.manual_seed(seed)

with open(bpe_path) as bpe_codes:
    bpe_model = BPE(bpe_codes)

def preprocess(line, is_source=True, source_lang=None, target_lang=None):
    return bpe_model.segment(line.lower())

def postprocess(line):
    return line.replace('@@ ', '')

def load_data(source_lang, target_lang, split='train', max_size=None):
    # max_size: max number of sentence pairs in the training corpus (None = all)
    path = os.path.join(data_dir, '{}.{}-{}'.format(split, *sorted([source_lang, target_lang])))
    return nmt_dataset.load_dataset(path, source_lang, target_lang, preprocess=preprocess, max_size=max_size)

train_data = load_data(source_lang, target_lang, 'train', max_size=max_size)   # set max_size to 10000 for fast debugging
valid_data = load_data(source_lang, target_lang, 'valid')
test_data = load_data(source_lang, target_lang, 'test')

source_dict_path = args.source_dict or os.path.join(model_dir, 'dict.{}.txt'.format(source_lang))
target_dict_path = args.target_dict or os.path.join(model_dir, 'dict.{}.txt'.format(target_lang))

source_dict = nmt_dataset.load_or_create_dictionary(
    source_dict_path,
    train_data['source_tokenized'],
    minimum_count=minimum_count,
    reset=True
)

target_dict = nmt_dataset.load_or_create_dictionary(
    target_dict_path,
    train_data['target_tokenized'],
    minimum_count=minimum_count,
    reset=True
)

print('source vocab size:', len(source_dict))
print('target vocab size:', len(target_dict))

nmt_dataset.binarize(train_data, source_dict, target_dict, sort=True)
nmt_dataset.binarize(valid_data, source_dict, target_dict, sort=False)
nmt_dataset.binarize(test_data, source_dict, target_dict, sort=False)

print('train_size={}, valid_size={}, test_size={}, min_len={}, max_len={}'.format(
    len(train_data),
    len(valid_data),
    len(test_data),
    train_data['source_len'].min(),
    train_data['source_len'].max(),
))

reset_seed()

train_iterator = nmt_dataset.BatchIterator(train_data, source_lang, target_lang, batch_size=batch_size, max_len=max_len, shuffle=True)
valid_iterator = nmt_dataset.BatchIterator(valid_data, source_lang, target_lang, batch_size=batch_size, max_len=max_len, shuffle=False)
test_iterator = nmt_dataset.BatchIterator(test_data, source_lang, target_lang, batch_size=batch_size, max_len=max_len, shuffle=False)

def save_model(model, checkpoint_path):
    dirname = os.path.dirname(checkpoint_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    torch.save(model, checkpoint_path)

def train_model(
        train_iterator,
        valid_iterators,
        model,
        checkpoint_path,
        epochs=10,
        validation_frequency=1
    ):
    """
    train_iterator: instance of nmt_dataset.BatchIterator or nmt_dataset.MultiBatchIterator
    valid_iterators: list of nmt_dataset.BatchIterator
    model: instance of nnet_models.EncoderDecoder
    checkpoint_path: path of the model checkpoint
    epochs: iterate this many times over train_iterator
    validation_frequency: validate the model every N epochs
    """

    reset_seed()

    best_bleu = -1
    for epoch in range(1, epochs + 1):

        start = time.time()
        running_loss = 0

        print('Epoch: [{}/{}]'.format(epoch, epochs))

        # Iterate over training batches for one epoch
        for i, batch in enumerate(train_iterator, 1):
            t = time.time()
            if verbose:
                sys.stdout.write(
                    "\r{}/{}, wall={:.2f}".format(
                        i,
                        len(train_iterator),
                        time.time() - start
                    )
                )
            running_loss += model.train_step(batch)

        if verbose:
            print()

        # Average training loss for this epoch
        epoch_loss = running_loss / len(train_iterator)

        print("loss={:.3f}, time={:.2f}".format(epoch_loss, time.time() - start))
        sys.stdout.flush()

        # Evaluate and save the model
        if epoch % validation_frequency == 0:
            bleu_scores = []
            
            # Compute BLEU over all validation sets
            for valid_iterator in valid_iterators:
                src, tgt = valid_iterator.source_lang, valid_iterator.target_lang
                translation_output = model.translate(valid_iterator, postprocess)
                bleu_score = translation_output.score
                output = translation_output.output

                with open(os.path.join(model_dir, 'valid.{}-{}.{}.out'.format(src, tgt, epoch)), 'w') as f:
                    f.writelines(line + '\n' for line in output)

                print('{}-{}: BLEU={}'.format(src, tgt, bleu_score))
                sys.stdout.flush()
                bleu_scores.append(bleu_score)

            # Average the validation BLEU scores
            bleu_score = round(sum(bleu_scores) / len(bleu_scores), 2)
            if len(bleu_scores) > 1:
                print('BLEU={}'.format(bleu_score))

            # Update the model's learning rate based on current performance.
            # This scheduler divides the learning rate by 10 if BLEU does not improve.
            model.scheduler_step(bleu_score)

            # Save a model checkpoint if it has the best validation BLEU so far
            if bleu_score > best_bleu:
                best_bleu = bleu_score
                save_model(model, checkpoint_path)

        print('=' * 50)

    print("Training completed. Best BLEU is {}".format(best_bleu))


encoder = nnet_models.TransformerEncoder(
    input_size=len(source_dict),
    hidden_size=hidden_size,
    num_layers=num_layers,
    dropout=dropout,
    heads=attention_heads
)
decoder = nnet_models.TransformerDecoder(
    output_size=len(target_dict),
    hidden_size=hidden_size,
    num_layers=num_layers,
    dropout=dropout,
    heads=attention_heads
)

model = nnet_models.EncoderDecoder(
    encoder,
    decoder,
    lr=learning_rate,
    use_cuda=not cpu,
    target_dict=target_dict
)

print('checkpoint path:', checkpoint_path)

train_model(train_iterator, [valid_iterator], model,
            epochs=epochs,
            checkpoint_path=checkpoint_path)
