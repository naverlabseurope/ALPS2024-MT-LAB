import numpy as np
import pandas as pd
import os
import functools
import torch
from torch import LongTensor
from typing import Optional, Any, Union, Callable, Iterator


SPECIAL_SYMBOLS = SOS_TOKEN, PAD_TOKEN, EOS_TOKEN, UNK_TOKEN = '<sos>', '<pad>', '<eos>', '<unk>'
# default special token ids used in fairseq models (e.g., NLLB):
SOS_IDX = EOS_IDX = 2
PAD_IDX = 1
UNK_IDX = 3


class Tokenizer:
    def __init__(self, model_path: str):
        import sentencepiece as spm
        self.model = spm.SentencePieceProcessor()
        self.model.Load(model_path)
    
    @functools.lru_cache(maxsize=10**4)  # to speed up tokenization of already-seen words
    def _tokenize(self, word: str) -> str:
        return ' '.join(self.model.encode_as_pieces(word))
    
    @functools.lru_cache(maxsize=10**6)  # to speed up tokenization of already-seen sentences
    def tokenize(self, line: str) -> str:
        line = line or ''  # to also work with None
        return ' '.join(self._tokenize(word) for word in line.split())
    
    def detokenize(self, line: str) -> str:
        return line.replace(' ', '').replace('▁', ' ').strip()


class Dictionary:
    def __init__(
        self,
        minimum_count: int = 10,
        unk_idx: int = UNK_IDX,
        sos_idx: int = SOS_IDX,
        eos_idx: int = EOS_IDX,
        pad_idx: int = PAD_IDX,
        shift: int = 4,  # fairseq-style
    ):
        self.words = []     # maps indices to words
        self.indices = {}   # maps words to indices
        self.counts = {}    # maps words to counts
        self.minimum_count = minimum_count
        self.unk_idx = unk_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.shift = shift

    def add_symbol(self, word: str, count: Optional[int] = None) -> None:
        count = count or self.minimum_count

        self.counts[word] = self.counts.get(word, 0) + count

        if word not in self.indices and self.counts[word] >= self.minimum_count:
            index = len(self.words) + self.shift
            self.words.append(word)
            self.indices[word] = index

    def __len__(self) -> int:
        return len(self.words) + self.shift

    def index(self, word: str) -> int:
        if word == EOS_TOKEN:
            return self.eos_idx
        elif word == SOS_TOKEN:
            return self.sos_idx
        elif word == UNK_TOKEN:
            return self.unk_idx
        elif word == PAD_TOKEN:
            return self.pad_idx
        else:
            return self.indices.get(word, self.unk_idx)
    
    def __getitem__(self, index: int) -> str:
        if index == self.eos_idx:
            return EOS_TOKEN
        elif index == self.pad_idx:
            return PAD_TOKEN
        elif index == self.sos_idx:
            return SOS_TOKEN
        elif index == self.unk_idx:
            return UNK_TOKEN
        else:
            return self.words[index - self.shift]

    def __setitem__(self, index: int, word: str) -> None:
        assert index not in (self.sos_idx, self.eos_idx, self.pad_idx, self.unk_idx)
        old_word = self.words[index - self.shift]
        self.words[index - self.shift] = word
        self.indices.pop(old_word)
        self.indices[word] = index

    def vec2txt(self, indices: Union[list[int], np.ndarray, LongTensor]) -> str:
        tokens = []
        for index in indices:
            if not isinstance(index, int):
                index = index.item()
            if index not in (self.sos_idx, self.eos_idx, self.pad_idx):  # skip special tokens
                tokens.append(self[index])
        return ' '.join(tokens)

    def txt2vec(self, sentence: str, add_eos: bool = False) -> np.ndarray:
        sentence = sentence or ''  # to work with None
        indices = [self.index(token) for token in sentence.split()]
        if add_eos:
            indices.append(self.eos_idx)
        return np.array(indices, dtype=np.int64)

    def save(self, path: str) -> None:
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(path, 'w') as f:
            f.writelines(
                "{} {}\n".format(word, self.counts[word]) for word in self.words
            )
    
    @staticmethod
    def load(path: str, minimum_count: int = 0) -> 'Dictionary':
        dictionary = Dictionary(minimum_count)

        with open(path, 'r') as f:
            for line in f:
                word, count = line.rsplit(' ', maxsplit=1)
                if word not in SPECIAL_SYMBOLS:
                    dictionary.add_symbol(word, int(count))
        return dictionary


def binarize(
    dataset: pd.DataFrame,
    source_dict: Dictionary,
    target_dict: Dictionary, sort: bool = True,
) -> bool:
    for key in 'source', 'target', 'prefix':
        dictionary = source_dict if key == 'source' else target_dict

        indices = []
        for tokens in dataset[key + '_tokenized']:
            indices.append(dictionary.txt2vec(tokens, add_eos=(key != 'prefix')))

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

    dataset.pad_idx = target_dict.pad_idx  # for collate()


def load_or_create_dictionary(dict_path: str, dataset: pd.DataFrame, reset: bool = False) -> Dictionary:
    if not reset and os.path.isfile(dict_path):
        dictionary = Dictionary.load(dict_path)
    else:
        dictionary = Dictionary()
        for line in dataset:
            for token in line.split():
                dictionary.add_symbol(token, count=1)
        dictionary.save(dict_path)

    return dictionary


def load_dataset(
    path: str,
    source_lang: Optional[str],
    target_lang: Optional[str],
    preprocess: Optional[Callable] = None,
    max_size: Optional[int] = None,
) -> pd.DataFrame:
    dataset = pd.DataFrame()

    def preprocess_and_split(source_line, target_line):
        if preprocess is not None:
            out = preprocess(
                source_line, target_line,
                source_lang=source_lang,
                target_lang=target_lang
            )
            
            if not out:
                return None
            
            source_line, target_line, *prefix = out
            # preprocess can return (source, target) or (source, target, prefix)
            prefix = prefix[0] if prefix else None
        
        return source_line, target_line, prefix

    with open(f'{path}.{source_lang}') as source_file, open(f'{path}.{target_lang}') as target_file:
        source_data = []
        target_data = []

        source_tokenized = []
        target_tokenized = []
        prefix_tokenized = []
        
        for source_line, target_line in zip(source_file, target_file):
            # if filter_fn is None or filter_fn(source_line, target_line):
            source_line, target_line = source_line.strip(), target_line.strip()
            tok_pair = preprocess_and_split(source_line, target_line)
            if not tok_pair:   # if 'preprocess' returns None, this means that we filter out this example
                continue
            src_tok, tgt_tok, prefix = tok_pair
            source_data.append(source_line)
            target_data.append(target_line)
            source_tokenized.append(src_tok)
            target_tokenized.append(tgt_tok)
            prefix_tokenized.append(prefix)

            if max_size and len(source_data) == max_size:
                break
        
        dataset['source_data'] = source_data
        dataset['target_data'] = target_data
        dataset['source_tokenized'] = source_tokenized
        dataset['target_tokenized'] = target_tokenized
        dataset['prefix_tokenized'] = prefix_tokenized

    return dataset


def concatenate_datasets(datasets: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(datasets, ignore_index=True)


class BatchIterator:
    def __init__(
        self,
        data: pd.DataFrame,
        source_lang: str,
        target_lang: str,
        batch_size: int,
        max_len: int,
        shuffle: bool = True,
    ):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.data = data
        pad_idx = data.pad_idx

        batches = []
        batch = []
        
        sample_size = 0
        
        for idx in range(len(data)):
            sample = {
                'source': data.iloc[idx]['source_bin'],
                'target': data.iloc[idx]['target_bin'],
                'reference': data.iloc[idx]['target_data'],
                'prefix': data.iloc[idx]['prefix_bin'],
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

        self.batches = [collate(batch, max_len, pad_idx) for batch in batches]
        self.shuffle = shuffle

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if self.shuffle:
            np.random.shuffle(self.batches)

        yield from self.batches


class MultilingualBatchIterator(BatchIterator):
    def __init__(self, iterators: list[BatchIterator], shuffle: bool = True):
        # Note that this builds homogeneous batches (all examples in a given batch are from the same language pair)
        # Heterogeneous batches might give better results
        self.iterators = iterators
        self.batches = sum((iterator.batches for iterator in iterators), [])
        self.shuffle = shuffle
        self.source_lang = 'src'
        self.target_lang = 'tgt'


def collate(batch: dict[str, Any], max_len: int, pad_idx: int) -> dict[str, Any]:
    # This function takes a batch containing samples of varying lengths and concatenates these samples 
    # into same length sequences by padding them to the maximum length
    empty_seq = np.array([], np.int64)
    # TODO: concatenate source and target here?
    source = [sample.get('source', empty_seq) for sample in batch]
    target = [sample.get('target', empty_seq) for sample in batch]
    prefix = [sample.get('prefix', empty_seq) for sample in batch]
    assert len(set(map(len, prefix))) == 1, 'all prefixes should have the same length'

    reference = [sample.get('reference') for sample in batch]
    max_source_len = min(max(map(len, source)), max_len)
    max_target_len = min(max(map(len, target)), max_len)

    def pad(seq: np.ndarray, max_len: int) -> tuple[np.ndarray, int]:
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
                mode="constant", constant_values=pad_idx,
            )
        return seq, seq_len

    source, source_len = zip(*[pad(x, max_source_len) for x in source])
    target, target_len = zip(*[pad(x, max_target_len) for x in target])
    
    batch = {
        'source': torch.tensor(np.array(source)),
        'target': torch.tensor(np.array(target)),
        'source_len': torch.tensor(np.array(source_len)),
        'target_len': torch.tensor(np.array(target_len)),
        'prefix': torch.tensor(np.array(prefix)),
        'reference': reference,
    }
    return batch
