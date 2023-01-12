import numpy as np
import pandas as pd
import os
import functools
import torch

SPECIAL_SYMBOLS = SOS_TOKEN, PAD_TOKEN, EOS_TOKEN, UNK_TOKEN = '<sos>', '<pad>', '<eos>', '<unk>'
SOS_IDX = EOS_IDX = 2    # like in fairseq
PAD_IDX = 1
UNK_IDX = 3


class Tokenizer:
    def __init__(self, model_path):
        import sentencepiece as spm
        self.model = spm.SentencePieceProcessor()
        self.model.Load(model_path)
    @functools.lru_cache(maxsize=10**4)  # to speed up tokenization of already seen words
    def _tokenize(self, word):
        return ' '.join(self.model.encode_as_pieces(word))
    @functools.lru_cache(maxsize=10**6)  # to speed up tokenization of already seen sentences
    def tokenize(self, line):
        return ' '.join(self._tokenize(word) for word in line.split())
    def detokenize(self, line):
        return line.replace(' ', '').replace('▁', ' ').strip()


class Dictionary:
    def __init__(self, minimum_count=10): # FIXME: why not 0 by default? 
        self.words = []     # maps indices to words
        self.indices = {}   # maps words to indices
        self.counts = {}    # maps words to counts
        self.minimum_count = minimum_count

        for token in SPECIAL_SYMBOLS:
            self.add_symbol(token)

    def add_symbol(self, word, count=None):
        count = count or self.minimum_count

        self.counts[word] = self.counts.get(word, 0) + count

        if word not in self.indices and self.counts[word] >= self.minimum_count:
            index = len(self.words)
            self.words.append(word)
            self.indices[word] = index

    def __len__(self):
        return len(self.words)

    def index(self, word):
        return self.indices.get(word, UNK_IDX)
    
    def __getitem__(self, index):
        return self.words[index]

    def __setitem__(self, index, word):
        old_word = self.words[index]
        self.words[index] = word
        self.indices.pop(old_word)
        self.indices[word] = index

    def vec2txt(self, indices):
        tokens = []
        for index in indices:
            if not isinstance(index, int):
                index = index.item()
            if index not in (EOS_IDX, SOS_IDX, PAD_IDX):
                tokens.append(self.words[index])
        return ' '.join(tokens)

    def txt2vec(self, sentence, add_eos=False):
        sentence = sentence or ''  # to work with None
        indices = [self.index(token) for token in sentence.split()]
        if add_eos:
            indices.append(EOS_IDX)
        return np.array(indices, dtype=np.int32)

    def save(self, path):
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(path, 'w') as f:
            f.writelines(
                "{} {}\n".format(word, self.counts[word]) for word in self.words
            )
    
    @staticmethod
    def load(path, minimum_count=10):
        dictionary = Dictionary(minimum_count)

        with open(path, 'r') as f:
            for line in f:
                word, count = line.rsplit(' ', maxsplit=1)
                if word not in SPECIAL_SYMBOLS:
                    dictionary.add_symbol(word, int(count))
        return dictionary


def binarize(dataset, source_dict, target_dict, sort=True):
    for key in 'source', 'target':
        dictionary = source_dict if key == 'source' else target_dict

        indices = []
        for tokens in dataset[key + '_tokenized']:
            indices.append(
                [dictionary.index(token) for token in tokens] + [EOS_IDX]
            )

        dataset[key + '_bin'] = indices
        dataset[key + '_len'] = dataset[key + '_bin'].apply(len) 

    dataset[:] = dataset[
        np.logical_and(
            dataset['source_len'] >= 2,
            dataset['target_len'] >= 2
        )
    ]

    if sort:
        dataset.sort_values(by=['source_len', 'target_len'], inplace=True, kind='mergesort')


def load_or_create_dictionary(dict_path, dataset, reset=False):
    if not reset and os.path.isfile(dict_path):
        dictionary = Dictionary.load(dict_path)
    else:
        dictionary = Dictionary()
        for tokens in dataset:
            for token in tokens:
                dictionary.add_symbol(token, count=1)
        dictionary.save(dict_path)

    return dictionary


def load_dataset(path, source_lang, target_lang, preprocess=None, max_size=None):
    dataset = pd.DataFrame()

    def preprocess_and_split(source_line, target_line):
        if preprocess is not None:
            tok_pair = preprocess(
                source_line, target_line,
                source_lang=source_lang,
                target_lang=target_lang
            )
            if not tok_pair:
                return None
            source_line, target_line = tok_pair
        return source_line.split(), target_line.split()

    with open(f'{path}.{source_lang}') as source_file, open(f'{path}.{target_lang}') as target_file:
        source_data = []
        target_data = []

        source_tokenized = []
        target_tokenized = []
        
        for source_line, target_line in zip(source_file, target_file):
            # if filter_fn is None or filter_fn(source_line, target_line):
            source_line, target_line = source_line.strip(), target_line.strip()
            tok_pair = preprocess_and_split(source_line, target_line)
            if not tok_pair:   # if 'preprocess' returns None, this means that we filter out this example
                continue
            src_tok, tgt_tok = tok_pair
            source_data.append(source_line)
            target_data.append(target_line)
            source_tokenized.append(src_tok)
            target_tokenized.append(tgt_tok)

            if max_size and len(source_data) == max_size:
                break
        
        dataset['source_data'] = source_data
        dataset['target_data'] = target_data
        dataset['source_tokenized'] = source_tokenized
        dataset['target_tokenized'] = target_tokenized

    return dataset


def concatenate_datasets(datasets):
    return pd.concat(datasets, ignore_index=True)


class BatchIterator:
    def __init__(self, data, source_lang, target_lang, batch_size, max_len, shuffle=True, prefix_len=0):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.data = data

        batches = []
        batch = []
        
        sample_size = 0
        
        for idx in range(len(data)):
            sample = {
                'source': data.iloc[idx]['source_bin'],
                'target': data.iloc[idx]['target_bin'],
                'reference': data.iloc[idx]['target_data'],
                'prefix': data.iloc[idx]['target_data'][:prefix_len],
            }

            size = max(len(sample['source']), len(sample['target']))
            
            if size > batch_size:
                continue

            sample_size = max(sample_size, size)

            if sample_size * (len(batch) + 1) > batch_size:
                batches.append(batch)
                batch = [sample]
                sample_size = size
            else:
                batch.append(sample)

        if batch:
            batches.append(batch)

        self.batches = [collate(batch, max_len, prefix_len=prefix_len) for batch in batches]
        self.shuffle = shuffle

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.batches)

        yield from self.batches


class MultilingualBatchIterator(BatchIterator):
    def __init__(self, iterators, shuffle=True):
        # Note that this builds homogeneous batches (all examples in a given batch are from the same language pair)
        # Heterogeneous batches might give better results
        self.iterators = iterators
        self.batches = sum((iterator.batches for iterator in iterators), [])
        self.shuffle = shuffle
        self.source_lang = 'src'
        self.target_lang = 'tgt'


def collate(batch, max_len, prefix_len=0):
    # This function takes a batch containing samples of varying lengths and concatenates these samples 
    # into same length sequences by padding them to the maximum length
    source = [sample['source'] for sample in batch]
    target = [sample['target'] for sample in batch]
    reference = [sample.get('reference') for sample in batch]
    max_source_len = min(max(map(len, source)), max_len)
    max_target_len = min(max(map(len, target)), max_len)

    def pad(seq, max_len):
        seq = np.array(seq)
        seq_len = len(seq)
        if seq_len > max_len:
            # truncate but keep the EOS token
            seq = np.concatenate([seq[:max_len - 1], seq[-1:]])
            seq_len = len(seq)
        elif seq_len < max_len:
            seq = np.pad(
                seq,
                pad_width=(0, max_len - seq_len),
                mode="constant", constant_values=PAD_IDX
            )
        return seq, seq_len

    source, source_len = zip(*[pad(x, max_source_len) for x in source])
    target, target_len = zip(*[pad(x, max_target_len) for x in target])
    
    batch = {
        'source': torch.tensor(np.array(source)),
        'target': torch.tensor(np.array(target)),
        'source_len': torch.tensor(np.array(source_len)),
        'target_len': torch.tensor(np.array(target_len)),
        'reference': reference,
    }
    if prefix_len > 0:
        batch['prefix'] = batch['target'][:,:prefix_len]
        assert batch['prefix'].shape[1] == prefix_len
    return batch
