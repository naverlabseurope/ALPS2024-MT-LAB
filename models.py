import os
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math
from collections import namedtuple

import sacrebleu

from data import PAD_IDX, SOS_IDX, EOS_IDX


class MultiheadAttention(nn.Module):
    """ Copied from fairseq """

    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.head_dim = embed_dim // num_heads
        assert (self.head_dim * num_heads == embed_dim), 'embed_dim must be divisible by num_heads'
        self.scaling = self.head_dim ** -0.5
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        tgt_len, batch_size, embed_dim = query.size()
        q = self.q_proj(query) * self.scaling
        q = q.contiguous().view(tgt_len, batch_size * self.num_heads, self.head_dim).transpose(0, 1)

        k = self.k_proj(key)
        k = k.contiguous().view(-1, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        v = self.v_proj(value)
        v = v.contiguous().view(-1, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        
        src_len = k.size(1)
        attn_weights = torch.bmm(q, k.transpose(1, 2))

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights.masked_fill_(attn_mask, float("-inf"))

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(batch_size, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float('-inf'),
            )
            attn_weights = attn_weights.view(batch_size * self.num_heads, tgt_len, src_len)

        attn_weights_float = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout(attn_weights)

        attn = torch.bmm(attn_probs, v)
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, batch_size, embed_dim)
        attn = self.out_proj(attn)

        attn_weights = attn_weights.view(batch_size, self.num_heads, tgt_len, src_len)
        return attn, torch.sum(attn_weights, dim=1) / self.num_heads, attn_weights


class BOW_Encoder(nn.Module):
    def __init__(self, input_size, hidden_size=512, reduce="max", num_layers=1, dropout=0, **kwargs):
        super(BOW_Encoder, self).__init__()
        self.emb_dim = hidden_size
        self.reduce = reduce
        assert(self.reduce in ["sum", "mean", "max"])
        self.hidden_size = hidden_size
        self.activation = nn.ReLU()
        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=PAD_IDX)
        self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList([nn.Linear(self.emb_dim, self.hidden_size)])
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(self.hidden_size, self.hidden_size))

    def forward(self, input, input_len, **kwargs):   # TODO: masking
        x = self.embedding(input)

        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
            x = self.dropout(x)
        
        mask = torch.arange(input.size(1))[None, :].to(input_len.device) < input_len[:, None]
        x = x * mask.unsqueeze(-1)

        if self.reduce == "mean":
            x = x.sum(dim=1) / input_len.unsqueeze(-1)
        elif self.reduce == "max":
            x = x.max(dim=1)[0]
        else:
            x = x.sum(dim=1)

        return x.unsqueeze(1), {}


class RNN_Encoder(nn.Module):
    """Encodes the input context."""

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0, **kwargs):
        """Initialize encoder.
        :param input_size: size of embedding
        :param hidden_size: size of GRU hidden layers
        :param num_layers: number of GRU layers
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=PAD_IDX)
        self.gru = nn.GRU(
            hidden_size, hidden_size, num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input, input_len, **kwargs):
        """Return encoded state.
        :param input: (batch_size x seqlen) tensor of token indices.
        """
        x = self.embedding(input)
        x = self.dropout(x)
        x = torch.nn.utils.rnn.pack_padded_sequence(x, input_len.cpu(), batch_first=True, enforce_sorted=False)
        x, _ = self.gru(x)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = self.dropout(x)
        return x, {}


class RNN_Decoder(nn.Module):
    """Generates a sequence of tokens in response to context."""

    def __init__(self, output_size, hidden_size, num_layers, dropout=0, **kwargs):
        
        """Initialize decoder.
        :param output_size: size of embedding
        :param hidden_size: size of GRU hidden layers
        :param num_layers: number of GRU layers
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size, padding_idx=PAD_IDX)
        self.dropout = nn.Dropout(p=dropout)
        self.gru = nn.GRU(
            hidden_size * 2, hidden_size, num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input, input_len, encoder_output, state=None, **kwargs):
        """Return encoded state.
        :param input: batch_size x tgt_len tensor of token indices.
        """
        bsz = input.size(0)
        tgt_len = input.size(1)
        
        x = self.embedding(input)       # BxTxH
        x = self.dropout(x)

        if encoder_output.size(1) == 1:
            y = encoder_output
        else:
            y = encoder_output[torch.arange(bsz), input_len -1].unsqueeze(1)
        y = y.repeat(1, tgt_len, 1)
        x = torch.cat([x, y], dim=-1)

        output, hidden = self.gru(x, state)
        # output: BxTxH
        scores = self.softmax(self.out(output))

        decoder_results = {
            'decoder_output': scores,
            'decoder_state': hidden,
        }
        return decoder_results


class AttentionModule(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(AttentionModule, self).__init__()
        self.l1 = nn.Linear(hidden_size, output_size, bias=False)
        self.l2 = nn.Linear(hidden_size + output_size, output_size, bias=False)

    def forward(self, hidden, encoder_outs, src_lens):
        """
        hidden: bsz x hidden_size
        encoder_outs: bsz x sq_len x output_size
        src_lens: bsz

        x: bsz x output_size
        attn_score: bsz x sq_len
        """
        x = self.l1(hidden)
        att_score = torch.bmm(encoder_outs, x.unsqueeze(-1))  # bsz x seq x 1
        att_score = att_score.squeeze(-1)       # bsz x seq
        att_score = att_score.transpose(0, 1)

        seq_mask = torch.arange(encoder_outs.size(1))[None, :].to(src_lens.device) < src_lens[:, None]
        seq_mask = seq_mask.transpose(0, 1)

        masked_att = seq_mask * att_score
        masked_att[masked_att == 0] = -1e10
        attn_scores = F.softmax(masked_att, dim=0)
        x = (
            attn_scores.unsqueeze(2) *
            encoder_outs.transpose(0, 1)
        ).sum(dim=0)
        x = torch.tanh(self.l2(torch.cat((x, hidden), dim=1)))
        return x, attn_scores


class AttentionDecoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers, dropout=0, **kwargs):
        super(AttentionDecoder, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size, padding_idx=PAD_IDX)
        self.dropout = nn.Dropout(p=dropout)
        self.gru = nn.GRU(
            hidden_size * 2, hidden_size, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0
        )
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.attention = AttentionModule(hidden_size, hidden_size)

    def forward(self, input, input_len, encoder_output, state=None, **kwargs):
        bsz = input.size(0)
        tgt_len = input.size(1)
        
        if state:
            hidden = state['hidden']
            context = state['context']
        else:
            hidden = None
            context = torch.zeros([bsz, self.hidden_size]).to(input.device)
        
        embed = self.embedding(input)
        embed = self.dropout(embed)
        embed = embed.transpose(0, 1)
        decoder_output = torch.empty(tgt_len, bsz, self.output_size).to(input.device)

        attn_weights = []

        for t in range(tgt_len):
            x = embed[t]
            x = torch.cat([x, context], dim=1)
            x, hidden = self.gru(x.unsqueeze(0), hidden)
            x = x.squeeze(0)
            context, attn_weights_ = self.attention(x, encoder_output, input_len)
            out = self.out(context)
            out = self.softmax(out)
            decoder_output[t] = out
            attn_weights.append(attn_weights_.transpose(0, 1))

        state = {'hidden': hidden, 'context': context}

        decoder_results = {
            'decoder_output': decoder_output.transpose(0, 1).contiguous(),
            'decoder_state': state,
            'attention_weights': torch.stack(attn_weights, axis=1),
        }
        return decoder_results


class TransformerEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0, heads=4, **kwargs):
        super(TransformerEncoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=PAD_IDX)
        self.embed_positions = PositionalEncoding(hidden_size)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_size, heads, hidden_size, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, input, input_len, **kwargs):
        x = self.embedding(input)
        mask = torch.arange(input.size(1))[None, :].to(input_len.device) >= input_len[:, None]
        # shape: (batch_size, src_len)

        x = x.transpose(0, 1)   # src_len first
        x += self.embed_positions(x)
        
        self_attn_weights = []
        for layer in self.layers:
            x, self_attn_weights_ = layer(
                x,
                src_key_padding_mask=mask
            )
            self_attn_weights.append(self_attn_weights_)

        meta = {'self_attention_weights': self_attn_weights}
        return x.transpose(0, 1), meta


class TransformerDecoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers=1, dropout=0, heads=4, **kwargs):
        super(TransformerDecoder, self).__init__()

        self.output_size = output_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size, padding_idx=PAD_IDX)
        self.embed_positions = PositionalEncoding(hidden_size)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(hidden_size, heads, hidden_size, dropout)
            for _ in range(num_layers)
        ])
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input, input_len, encoder_output, state=None, **kwargs):
        """
        input: Tensor of shape (batch_size, tgt_len)
        input_len: Tensor of shape (batch_size,)
        encoder_output: Tensor of shape (batch_size, src_len, hidden_size)
        """
        x = self.embedding(input)
        size = encoder_output.size(1)
        memory_mask = torch.arange(size)[None, :].to(input_len.device) >= input_len[:, None]
        # shape: (batch_size, src_len)

        if not self.training:
            tgt_mask = None
        else:
            size = input.size(1)
            tgt_mask = (torch.triu(torch.ones(size, size)) != 1).transpose(0, 1)
            tgt_mask = tgt_mask.to(input_len.device)   # shape: (tgt_len, tgt_len)

        x = x.transpose(0, 1)   # src_len first
        encoder_output = encoder_output.transpose(0, 1)   # src_len first
        
        if not self.training and state is not None:
            start = state[0].size(0)
        else:
            start = 0
        x += self.embed_positions(x, start)
        
        if state is None:
            state = [None] * len(self.layers)

        self_attn_weights = []

        for i, layer in enumerate(self.layers):
            if state[i] is None:
                state[i] = x
            else:
                state[i] = torch.cat([state[i], x], dim=0)

            x, self_attn_weights_, attn_weights = layer(
                x,
                encoder_output,
                memory_key_padding_mask=memory_mask,
                tgt_mask=tgt_mask,
                prev_states=state[i]
            )
            if self.training:
                attn_weights = self_attn_weights_ = None

            self_attn_weights.append(self_attn_weights_)

        scores = self.softmax(self.out(x.transpose(0, 1)))

        decoder_results = {
            'decoder_output': scores,
            'decoder_state': state,
            'attention_weights': attn_weights,
            'self_attention_weights': self_attn_weights
        }
        return decoder_results


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, target_dict, lr=1e-3, use_cuda=True, max_len=50, clip=0.3):
        super(EncoderDecoder, self).__init__()

        self.device = 'cuda' if (torch.cuda.is_available() and use_cuda) else 'cpu'

        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)

        self.target_dict = target_dict

        # set up the criterion
        self.criterion = nn.NLLLoss(reduction='sum', ignore_index=PAD_IDX)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='max', patience=0, factor=0.1,
            threshold=0.5, threshold_mode='abs', verbose=True
        )
    
        self.max_len = max_len
        self.clip = clip
        self.START = torch.LongTensor([SOS_IDX]).to(self.device)

        self.metrics = {}

    def vec2txt(self, vector):
        """
        Convert vector to text.
        :param vector: tensor of token indices.
        1-d tensors will return a string, 2-d will return a list of strings
        """
        if vector.dim() == 1:
            output_tokens = []
            # Remove the final END_TOKEN that is appended to predictions
            for token in vector:
                if token == EOS_IDX:
                    break
                else:
                    output_tokens.append(token)
            return self.target_dict.vec2txt(output_tokens)

        elif vector.dim() == 2:
            return [self.vec2txt(vector[i]) for i in range(vector.size(0))]
        raise RuntimeError(
            "Improper input to vec2txt with dimensions {}".format(vector.size())
        )

    def translate(self, batch_iterator, detokenize=None):
        hypotheses = []
        references = []

        for data in batch_iterator:
            hypotheses += self.decoding_step(data)[0]
            references += data['reference']

        if detokenize is not None:
            hypotheses = [detokenize(line) for line in hypotheses]

        chrf = sacrebleu.corpus_chrf(hypotheses, [references])
        translation_output = namedtuple('translation_output', ['score', 'output'])
        return translation_output(round(chrf.score, 2), hypotheses)

    def train_step(self, batch, train=True):
        input, input_len, target, target_len = batch['source'], batch['source_len'], batch['target'], batch['target_len']

        if input is None:
            return

        input = input.to(self.device)
        target = target.to(self.device)
        input_len = input_len.to(self.device)
        target_len = target_len.to(self.device)

        bsz = input.size(0)
        start = self.START.expand(bsz, 1)
        loss = 0
        if train:
            self.zero_grad()
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()

        encoder_output, _ = self.encoder(input, input_len=input_len)

        # Teacher forcing: Feed the target as the next input
        shifted_target = target.narrow(1, 0, target.size(1) - 1)
        decoder_input = torch.cat([start, shifted_target], 1)

        decoder_results = self.decoder(
            input=decoder_input,
            input_len=input_len,
            encoder_output=encoder_output,
        )
        decoder_output = decoder_results['decoder_output']
        scores = decoder_output.view(-1, decoder_output.size(-1))
        loss = self.criterion(scores, target.view(-1)) / target_len.sum()
        
        if train:
            loss.backward()
            self.update_params()

        return loss.item()

    def eval_step(self, batch):
        return self.train_step(batch, train=False)

    def zero_grad(self):
        """Zero out optimizer."""
        self.optimizer.zero_grad()

    def update_params(self):
        """Do one optimization step."""
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.encoder.parameters(), self.clip)
            torch.nn.utils.clip_grad_norm_(
                self.decoder.parameters(), self.clip)
        self.optimizer.step()

    def scheduler_step(self, val_score=None):
        if isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(val_score)
        else:
            self.scheduler.step()

    def decoding_step(self, batch):
        input, input_len = batch['source'], batch['source_len']

        if input is None:
            return

        input = input.to(self.device)
        input_len = input_len.to(self.device)

        bsz = input.size(0)
        starts = self.START.expand(bsz, 1)  # expand to batch size
        self.encoder.eval()
        self.decoder.eval()
        encoder_output, meta = self.encoder(input, input_len=input_len)
        encoder_self_attn = meta.get('self_attention_weights')

        predictions = []
        done = [False for _ in range(bsz)]
        total_done = 0
        decoder_input = starts
        decoder_state = None
        attn_weights = []

        for _ in range(self.max_len):
            # generate at most max_len tokens
            decoder_results = self.decoder(
                input=decoder_input,
                input_len=input_len,
                encoder_output=encoder_output,
                state=decoder_state,
            )

            decoder_output = decoder_results['decoder_output']
            decoder_state = decoder_results.get('decoder_state')
            attn_weights_ = decoder_results.get('attention_weights')

            _, preds = decoder_output.max(2)
            predictions.append(preds)
            decoder_input = preds  # set input to next step

            attn_weights.append(attn_weights_)

            # check if we've produced the end token
            for b in range(bsz):
                if not done[b]:
                    # only add more tokens for examples that aren't done
                    if preds[b].item() == EOS_IDX:
                        # if we produced END, we're done
                        done[b] = True
                        total_done += 1
            if total_done == bsz:
                # no need to generate any more
                break
        predictions = torch.cat(predictions, 1)
        
        attn = None if attn_weights[0] is None else torch.cat(attn_weights, 1)
        return self.vec2txt(predictions), attn, encoder_self_attn

    def load(self, path):
        if os.path.isfile(path):
            ckpt = torch.load(path)
            self.load_state_dict(ckpt['model'])
            if ckpt.get('scheduler'):
                self.scheduler.load_state_dict(ckpt['scheduler'])
            if ckpt.get('optimizer'):
                self.optimizer.load_state_dict(ckpt['optimizer'])
            if ckpt.get('metrics'):
                self.metrics = ckpt['metrics']

    def save(self, path):
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        ckpt = {
            'model': self.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'metrics': self.metrics,
        }
        torch.save(ckpt, path)

    def record(self, name, value):
        self.metrics.setdefault(self.epoch, {})[name] = value

    @property
    def epoch(self):
        return self.scheduler.last_epoch


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, ffn_dim, dropout=0, activation=F.relu):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(hidden_size, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(hidden_size, ffn_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ffn_dim, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        x = src
        residual = x
        x, _, self_attn_weights_all_heads = self.self_attn(
            x, x, x, attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask)
        x = residual + self.dropout1(x)
        x = self.norm1(x)
        residual = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = residual + self.dropout2(x)
        x = self.norm2(x)
        return x, self_attn_weights_all_heads


class TransformerDecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, ffn_dim, dropout=0, activation=F.relu):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(hidden_size, num_heads, dropout=dropout)
        self.multihead_attn = MultiheadAttention(hidden_size, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(hidden_size, ffn_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ffn_dim, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None, prev_states=None):
        x = tgt
        residual = x
        if prev_states is None:
            prev_states = x
        x, _, self_attn_weights_all_heads = self.self_attn(
            x, prev_states, prev_states, attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask)
        x = residual + self.dropout1(x)
        x = self.norm1(x)

        residual = x
        x, attn_weights, attn_weights_all_heads = self.multihead_attn(
            x, memory, memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)
        x = residual + self.dropout2(x)
        x = self.norm2(x)

        residual = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = residual + self.dropout3(x)
        x = self.norm3(x)

        return x, self_attn_weights_all_heads, attn_weights


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0).transpose(0, 1)

    def forward(self, x, start=0):
        return x + self.pe[start:start + x.size(0), :].to(x.device)
