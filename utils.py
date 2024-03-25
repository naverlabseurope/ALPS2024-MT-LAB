import torch
import time
import statistics
import matplotlib.pyplot as plt
import numpy as np
import os
from models import EncoderDecoder
from data import EOS_TOKEN
from contextlib import contextmanager


@contextmanager
def benchmark():
    torch.cuda.empty_cache()
    mem = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()    
    start = time.time()
    yield
    elapsed = time.time() - start
    mem = torch.cuda.max_memory_allocated() - mem
    print(f'Time elapsed: {elapsed:.1f} sec, GPU memory usage: {mem / 2**20:.1f}MiB')


def free_gpu_memory():
    """ Move all models to the CPU to free GPU memory """
    global_variables = globals()
    for k, v in global_variables.items():
        if isinstance(v, EncoderDecoder):
            v.cpu()
    torch.cuda.empty_cache()


def plot_attention(input: str, output: str, attention_weights: np.ndarray):
    """
    Plot an encoder-decoder attention matrix
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.colorbar(ax.matshow(attention_weights, cmap='bone', aspect='auto'))
    xlabels = input.split() + [EOS_TOKEN]
    ylabels = output.split() + [EOS_TOKEN]
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=90)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    plt.show()


def plot_loss(model):
    """
    Plot the training VS validation loss and chrf for the given model
    (provided those metrics are stored in the checkpoint)
    """
    metrics = model.metrics
    epochs = sorted(metrics.keys())
    train_loss = [metrics[epoch]['train_loss'] for epoch in epochs]
    valid_loss = [
        statistics.mean(v for k, v in metrics[epoch].items() if 'loss' in k and k != 'train_loss')
        for epoch in epochs
    ]
    chrf = [
        statistics.mean(v for k, v in metrics[epoch].items() if 'chrf' in k)
        for epoch in epochs
    ]
    
    _, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(epochs, train_loss, linestyle='solid', label='Train loss')
    ax1.plot(epochs, valid_loss, linestyle='dashdot', label='Valid loss')
    ax2.plot(epochs, chrf, 'g--', label='Valid chrF')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax2.set_ylabel('chrF')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
