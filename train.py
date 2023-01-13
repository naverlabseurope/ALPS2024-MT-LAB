#!env/bin/python3

import os
import sys
import models
import data
import numpy as np
import torch
import time
import json
import sacrebleu
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('checkpoint_path', help='Path to the output model checkpoint (e.g., models/my_model.pt)')
parser.add_argument('--reset', action='store_true', help='if the checkpoint already exists, reset it instead of resuming training')
parser.add_argument('-s', '--source-lang', default='en', metavar='SRC', help='Source language (e.g., en)')
parser.add_argument('-t', '--target-lang', default='fr', metavar='TGT', help='Target language (e.g., fr)')
parser.add_argument('--lang-pairs', nargs='+',
                    help='Manually define a list of language pairs (like this: de-en en-fr) to train multilingual models. '
                    '--source-lang and --target-lang will be ignored, and the dictionaries will be saved as dict.src.txt and dict.tgt.txt. '
                    'If there are more than one target language, source-side language codes will automatically be added.')
parser.add_argument('--source-dict', help='Path to the source dictionary. Default: dict.SRC.txt in the same directory as the checkpoint')
parser.add_argument('--target-dict', help='Path to the target dictionary. Default: dict.TGT.txt in the same directory as the checkpoint')
parser.add_argument('--bpe-path', help='Path to the BPE model. Default: DATA_DIR/spm.de-en-fr.model')
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs. Default: 10')
parser.add_argument('--encoder-layers', type=int, default=1, help='Number of encoder layers. Default: 1')
parser.add_argument('--decoder-layers', type=int, default=1, help='Number of decoder layers. Default: 1')
parser.add_argument('--embed-dim', type=int, default=512, help='Embedding dimension of the model. Default: 512')
parser.add_argument('--ffn-dim', type=int, help='Feed-forward dimension of the Transformer, if different than --embed-dim')
parser.add_argument('--max-size', type=int, help='Maximum number of training examples. Default: all')
parser.add_argument('--max-valid-size', type=int, default=500, help='Maximum number of validation examples. Default: 500')
parser.add_argument('--max-len', type=int, default=30, help='Maximum number of tokens per line (longer sequences will be truncated). Default: 30')
parser.add_argument('--data-dir', default='data', metavar='DATA_DIR', help='Directory containing the training data. Default: data')
parser.add_argument('--dropout', type=float, default=0, help='Dropout rate. Default: 0')
parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate. Default: 0.0005')
parser.add_argument('--heads', type=int, default=4, help='Number of attention heads. Default: 4')
parser.add_argument('--batch-size', type=int, default=512, help='Maximum number of tokens in a batch. Default: 512')
parser.add_argument('--cpu', action='store_true', help='Train on CPU')
parser.add_argument('-v', '--verbose', action='store_true', help='Show training progress on a step-by-step basis')
parser.add_argument('--shared-embeddings', action='store_true',
                    help='Use the same joint dictionary on the source and target side, '
                    ' and share the encoder and decoder embedding matrices')
parser.add_argument('--model-type', choices=['bow', 'rnn', 'transformer'], default='transformer',
                    help='which model architecture to use')
parser.add_argument('--warmup', type=int, help='Use an inverse square root warmup schedule with this number of warmup '
                    'steps. Default: Reduce LR on plateau')
parser.add_argument('--scheduler-args', help='Serialized json dict containing additional arguments for the scheduler')
parser.add_argument('--label-smoothing', type=float, default=0, help='Amount of label smoothing. Default: 0')

args = parser.parse_args()
print(args)

model_dir = os.path.dirname(args.checkpoint_path)
bpe_path = args.bpe_path or os.path.join(args.data_dir, 'spm.de-en-fr.model')

if args.lang_pairs:
    lang_pairs = [tuple(lang_pair.split('-')) for lang_pair in args.lang_pairs]
else:
    lang_pairs = [(args.source_lang, args.target_lang)]

if len(lang_pairs) == 1:
    source_lang, target_lang = lang_pairs[0]
else:
    source_lang, target_lang = 'src', 'tgt'

lang_pairs.sort()

source_langs = set(src for src, _ in lang_pairs)
target_langs = set(tgt for _, tgt in lang_pairs)

source_dict_path = (
    args.source_dict or
    os.path.join(model_dir, 'dict.txt' if args.shared_embeddings else f'dict.{source_lang}.txt')
)
target_dict_path = (
    args.target_dict or
    os.path.join(model_dir, 'dict.txt' if args.shared_embeddings else f'dict.{target_lang}.txt')
)


def reset_seed(seed=1234):
    np.random.seed(seed)
    torch.manual_seed(seed)


tokenizer = data.Tokenizer(bpe_path)


def preprocess(source_line, target_line, source_lang=None, target_lang=None):
    source_line = tokenizer.tokenize(source_line)
    target_line = tokenizer.tokenize(target_line)

    if len(target_langs) > 1:
        source_line = f'<lang:{target_lang}> {source_line}'
    return source_line, target_line


def postprocess(line):
    return tokenizer.detokenize(line)


def load_data(source_lang, target_lang, split='train', max_size=None):
    # max_size: max number of sentence pairs in the training corpus (None = all)
    path = os.path.join(args.data_dir, f'{split}.{source_lang}-{target_lang}')
    return data.load_dataset(path, source_lang, target_lang, preprocess=preprocess, max_size=max_size)


train_data = {}
valid_data = {}

source_data = []
target_data = []

for lang_pair in lang_pairs:
    src, tgt = lang_pair
    train_data[lang_pair] = load_data(src, tgt, 'train', max_size=args.max_size)   # set max_size to 10000 for fast debugging
    valid_data[lang_pair] = load_data(src, tgt, 'valid', max_size=args.max_valid_size)
    source_data += list(train_data[lang_pair]['source_tokenized'])
    target_data += list(train_data[lang_pair]['target_tokenized'])

if args.shared_embeddings:
    source_data += target_data

source_dict = data.load_or_create_dictionary(
    source_dict_path,
    source_data,
    reset=args.reset,
)

if args.shared_embeddings:
    target_dict = source_dict    
else:
    target_dict = data.load_or_create_dictionary(
        target_dict_path,
        target_data,
        reset=args.reset,
    )

print('source vocab size:', len(source_dict))
print('target vocab size:', len(target_dict))

train_iterators = []
valid_iterators = []

for lang_pair in lang_pairs:
    src, tgt = lang_pair
    data.binarize(train_data[lang_pair], source_dict, target_dict, sort=True)
    data.binarize(valid_data[lang_pair], source_dict, target_dict, sort=False)

    print('{}-{}: train_size={}, valid_size={}, min_len={}, max_len={}, avg_len={:.1f}'.format(
        src,
        tgt,
        len(train_data[lang_pair]),
        len(valid_data[lang_pair]),
        train_data[lang_pair]['source_len'].min(),
        train_data[lang_pair]['source_len'].max(),
        train_data[lang_pair]['source_len'].mean(),
    ))

    reset_seed()

    train_iterator = data.BatchIterator(train_data[lang_pair], src, tgt, batch_size=args.batch_size, max_len=args.max_len, shuffle=True)
    train_iterators.append(train_iterator)
    valid_iterator = data.BatchIterator(valid_data[lang_pair], src, tgt, batch_size=args.batch_size, max_len=args.max_len, shuffle=False)
    valid_iterators.append(valid_iterator)

if len(train_iterator) > 1:
    train_iterator = data.MultilingualBatchIterator(train_iterators)
else:
    train_iterator = train_iterators[0]


def evaluate_model(model, *test_or_valid_iterators, record=False):
    """
    Evaluate given models with given test or validation sets. This will compute both chrF and validation loss.
    
    model: instance of models.EncoderDecoder
    test_or_valid_iterators: list of BatchIterator
    record: save scores in the model checkpoint
    """
    scores = []
    
    model.half()  # half-precision decoding is faster on some GPUs (i.e., model parameters and activations
    # are stored in float16 format instead of float32)
    
    # Compute chrF and valid loss over all test or validation sets
    for iterator in test_or_valid_iterators:
        loss = 0
        hypotheses = []
        references = []
        
        for batch in iterator:
            loss += model.eval_step(batch) / len(iterator)
            hyps, _ = model.translate(batch)
            hypotheses += [postprocess(hyp) for hyp in hyps]  # detokenize
            references += batch['reference']
        
        chrf = sacrebleu.corpus_chrf(hypotheses, [references]).score

        src, tgt = iterator.source_lang, iterator.target_lang
        print(f'{src}-{tgt}: loss={loss:.2f}, chrF={chrf:.2f}')
        if record:  # store the metrics in the model checkpoint
            model.record(f'{src}_{tgt}_loss', loss)
            model.record(f'{src}_{tgt}_chrf', chrf)
        
        scores.append(chrf)

    # Average the validation chrF scores
    score = sum(scores) / len(scores)
    return score


def train_model(model, train_iterator, valid_iterators, checkpoint_path, epochs=10):
    """
    model: instance of models.EncoderDecoder
    train_iterator: instance of data.BatchIterator or data.MultiBatchIterator
    valid_iterators: list of data.BatchIterator
    checkpoint_path: path of the model checkpoint
    epochs: iterate this many times over train_iterator
    """

    reset_seed()

    if model.epoch > epochs:
        evaluate_model(model, *valid_iterators, record=False)
        return
    
    start = time.time()

    best_score = -1
    while model.epoch <= epochs:
        model.float()

        start_ = time.time()
        running_loss = 0

        print(f'Epoch [{model.epoch}/{epochs}]')

        # Iterate over training batches for one epoch
        for i, batch in enumerate(train_iterator, 1):
            running_loss += model.train_step(batch)
            model.scheduler_step(end_of_epoch=False)
            if args.verbose:
                sys.stderr.write(
                    "\r{}/{}, wall={:.2f}, train_loss={:.3f}, lr={:.4e}".format(
                        i,
                        len(train_iterator),
                        time.time() - start_,
                        running_loss / i,
                        model.optimizer.param_groups[0]['lr'],
                    )
                )

        if args.verbose:
            sys.stderr.write('\n')

        # Average training loss for this epoch
        epoch_loss = running_loss / len(train_iterator)

        print(f"train_loss={epoch_loss:.3f}, time={time.time() - start_:.2f}")
        model.record('train_loss', epoch_loss)

        score = evaluate_model(model, *valid_iterators, record=True)

        # Update the model's learning rate based on current performance.
        # This scheduler divides the learning rate by 10 if chrF does not improve.
        model.scheduler_step(score=score, end_of_epoch=True)

        # Save a model checkpoint if it has the best validation chrF so far
        if score > best_score:
            best_score = score
            model.save(checkpoint_path)

        print('=' * 50, flush=True)

    print(f'Training completed. Best chrF is {best_score:.2f}. Total time: {time.time() - start:.2f}')


encoder_args = dict(
    source_dict=source_dict,
    embed_dim=args.embed_dim,
    num_layers=args.encoder_layers,
    dropout=args.dropout,
    ffn_dim=args.ffn_dim or args.embed_dim,
)
decoder_args = dict(
    target_dict=target_dict,
    embed_dim=args.embed_dim,
    num_layers=args.decoder_layers,
    dropout=args.dropout,
    ffn_dim=args.ffn_dim or args.embed_dim,
)

if args.model_type == 'bow':
    encoder = models.BOW_Encoder(**encoder_args)
    decoder = models.RNN_Decoder(**decoder_args)
elif args.model_type == 'rnn':
    encoder = models.RNN_Encoder(**encoder_args)
    decoder = models.RNN_Decoder(**decoder_args)
else:
    encoder = models.TransformerEncoder(**encoder_args, heads=args.heads)
    decoder = models.TransformerDecoder(**decoder_args, heads=args.heads)

if args.shared_embeddings:
    decoder.embed_tokens = encoder.embed_tokens

scheduler_args = json.loads(args.scheduler_args) if args.scheduler_args else {}
if args.warmup:
    scheduler_args['warmup'] = args.warmup

model = models.EncoderDecoder(
    encoder,
    decoder,
    lr=args.lr,
    label_smoothing=args.label_smoothing,
    use_cuda=not args.cpu,
    scheduler=models.WarmupLR if args.warmup else None,
    scheduler_args=scheduler_args,
    clip=1.0,
)

num_params = sum(p.numel() for p in model.parameters())
print(f'total model parameters: {num_params}')

if not args.reset:
    model.load(args.checkpoint_path)

if model.epoch > 1:
    print(f'resumed checkpoint {args.checkpoint_path} ({model.epoch - 1} epochs)')
else:
    print(f'new model checkpoint: {args.checkpoint_path}')

try:
    train_model(model, train_iterator, valid_iterators,
                epochs=args.epochs,
                checkpoint_path=args.checkpoint_path)
except KeyboardInterrupt:
    print('training interrupted')
