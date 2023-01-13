import os
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math
import functools
from contextlib import contextmanager

from data import PAD_IDX, SOS_IDX, EOS_IDX


def checkpoint(module):
    module._orig_forward = module.forward
    from torch.utils.checkpoint import checkpoint
    module.forward = functools.partial(checkpoint, module._orig_forward, use_reentrant=False)
    return module


class Encoder(nn.Module):
    def __init__(self, source_dict, embed_dim=512, num_layers=1, dropout=0, **kwargs):
        super().__init__()
        self.source_dict = source_dict
        self.input_size = len(source_dict)
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.embed_tokens = nn.Embedding(self.input_size, self.embed_dim, padding_idx=PAD_IDX)
        nn.init.normal_(self.embed_tokens.weight, mean=0, std=self.embed_tokens.weight.size(-1) ** -0.5)
        nn.init.constant_(self.embed_tokens.weight[PAD_IDX], 0)
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def select_adapter(self, id):
        pass


class Decoder(nn.Module):
    def __init__(self, target_dict, embed_dim=512, num_layers=1, dropout=0, embed_tokens=None, **kwargs):
        super().__init__()
        self.target_dict = target_dict
        self.output_size = len(target_dict)
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        if embed_tokens is None:
            self.embed_tokens = nn.Embedding(self.output_size, self.embed_dim, padding_idx=PAD_IDX)
            nn.init.normal_(self.embed_tokens.weight, mean=0, std=self.embed_tokens.weight.size(-1) ** -0.5)
            nn.init.constant_(self.embed_tokens.weight[PAD_IDX], 0)
        else:
            self.embed_tokens = embed_tokens
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def select_adapter(self, id):
        pass


class MultiheadAttention(nn.Module):
    """ Copied from fairseq """

    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
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
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)
        self.reset_state()

    def reset_state(self):
        self.state = None

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None, incremental=False):
        tgt_len, batch_size, embed_dim = query.size()
        q = self.q_proj(query) * self.scaling
        q = q.contiguous().view(tgt_len, batch_size * self.num_heads, self.head_dim).transpose(0, 1)

        k = self.k_proj(key)
        k = k.contiguous().view(-1, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        v = self.v_proj(value)
        v = v.contiguous().view(-1, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        
        if incremental and self.state is not None:
            prev_key = self.state['prev_key']
            prev_value = self.state['prev_value']
            prev_key_padding_mask = self.state['prev_key_padding_mask']
            prev_key = prev_key.view(batch_size * self.num_heads, -1, self.head_dim)
            prev_value = prev_value.view(batch_size * self.num_heads, -1, self.head_dim)
            k = torch.cat([prev_key, k], dim=1)
            v = torch.cat([prev_value, v], dim=1)
            key_padding_mask = torch.cat([prev_key_padding_mask.float(), key_padding_mask.float()], dim=1)

        if incremental:
            self.state = {
                'prev_key': k.view(batch_size, self.num_heads, -1, self.head_dim),
                'prev_value': v.view(batch_size, self.num_heads, -1, self.head_dim),
                'prev_key_padding_mask': key_padding_mask,
            }

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


class BOW_Encoder(Encoder):
    def __init__(self, *args, reduce="max", **kwargs):
        super().__init__(*args, **kwargs)
        self.reduce = reduce
        assert(self.reduce in ["sum", "mean", "max"])
        self.activation = nn.ReLU()
        self.layers = nn.ModuleList([nn.Linear(self.embed_dim, self.embed_dim)])
        for _ in range(self.num_layers - 1):
            self.layers.append(nn.Linear(self.embed_dim, self.embed_dim))

    def forward(self, input, input_len, **kwargs):
        x = self.embed_tokens(input)

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

        return x.unsqueeze(1)


class RNN_Encoder(Encoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gru = nn.GRU(
            self.embed_dim, self.embed_dim, num_layers=self.num_layers, batch_first=True,
            dropout=self.dropout_rate if self.num_layers > 1 else 0
        )

    def forward(self, input, input_len, **kwargs):
        """Return encoded state.
        :param input: (batch_size x seqlen) tensor of token indices.
        """
        x = self.embed_tokens(input)
        x = self.dropout(x)
        x = torch.nn.utils.rnn.pack_padded_sequence(x, input_len.cpu(), batch_first=True, enforce_sorted=False)
        x, _ = self.gru(x)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = self.dropout(x)
        return x


class RNN_Decoder(Decoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gru = nn.GRU(
            self.embed_dim * 2, self.embed_dim, num_layers=self.num_layers, batch_first=True,
            dropout=self.dropout_rate if self.num_layers > 1 else 0
        )
        self.out = nn.Linear(self.embed_dim, self.output_size)
        self.reset_state()

    def reset_state(self):
        self.state = None

    def forward(self, input, input_len, encoder_output, incremental=False, **kwargs):
        """Return encoded state.
        :param input: batch_size x tgt_len tensor of token indices.
        """
        bsz = input.size(0)
        tgt_len = input.size(1)
        
        x = self.embed_tokens(input)       # BxTxH
        x = self.dropout(x)

        if encoder_output.size(1) == 1:
            y = encoder_output
        else:
            y = encoder_output[torch.arange(bsz), input_len -1].unsqueeze(1)
        y = y.repeat(1, tgt_len, 1)
        x = torch.cat([x, y], dim=-1)

        output, hidden = self.gru(x, self.state)
        if incremental:
            self.state = hidden
        # output: BxTxH
        x = self.out(output)
        return x, None


class TransformerEncoder(Encoder):
    def __init__(self, *args, heads=4, ffn_dim=None, checkpointing=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.heads = heads
        self.ffn_dim = ffn_dim or self.embed_dim
        self.embed_positions = PositionalEncoding(self.embed_dim)
        self.layers = nn.ModuleList([self.build_layer(i) for i in range(self.num_layers)])
        if checkpointing:
            for layer in self.layers:
                checkpoint(layer)
        self.layer_norm = nn.LayerNorm(self.embed_dim)
        self.embed_scale = math.sqrt(self.embed_dim)
        self.dropout = nn.Dropout(self.dropout_rate)
    
    def build_layer(self, layer_id):
        return TransformerEncoderLayer(self.embed_dim, self.heads, self.dropout_rate, self.ffn_dim)

    def forward(self, input, input_len, **kwargs):
        x = self.embed_tokens(input) * self.embed_scale

        mask = torch.arange(input.size(1))[None, :].to(input_len.device) >= input_len[:, None]
        # shape: (batch_size, src_len)

        x = x.transpose(0, 1)   # src_len first
        x += self.embed_positions(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, src_key_padding_mask=mask)

        x = self.layer_norm(x)
        return x.transpose(0, 1)


class TransformerDecoder(Decoder):
    def __init__(self, *args, heads=4, ffn_dim=None, checkpointing=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.heads = heads
        self.ffn_dim = ffn_dim or self.embed_dim
        self.embed_positions = PositionalEncoding(self.embed_dim)
        self.layers = nn.ModuleList([self.build_layer(i) for i in range(self.num_layers)])
        if checkpointing:
            for layer in self.layers:
                checkpoint(layer)
        self.layer_norm = nn.LayerNorm(self.embed_dim)
        self.embed_scale = math.sqrt(self.embed_dim)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.future_mask = torch.empty(0, dtype=torch.bool)

    def build_layer(self, layer_id):
        return TransformerDecoderLayer(self.embed_dim, self.heads, self.dropout_rate, self.ffn_dim)

    def reset_state(self):
        self.embed_positions.reset_state()
        for layer in self.layers:
            layer.self_attn.reset_state()

    def forward(self, input, input_len, encoder_output, incremental=False, **kwargs):
        """
        input: Tensor of shape (batch_size, tgt_len)
        input_len: Tensor of shape (batch_size,)
        encoder_output: Tensor of shape (batch_size, src_len, embed_dim)
        """
        x = self.embed_tokens(input) * self.embed_scale
        src_len = encoder_output.size(1)
        encoder_mask = torch.arange(src_len)[None, :].to(input_len.device) >= input_len[:, None]
        padding_mask = input.eq(PAD_IDX)
        # shape: (batch_size, src_len)

        tgt_len = input.size(1)
        if self.future_mask.size(0) < tgt_len:
            self.future_mask = (torch.triu(torch.ones(tgt_len, tgt_len)) != 1).transpose(0, 1)
        self.future_mask = self.future_mask.to(x.device)
        future_mask = self.future_mask[:tgt_len, :tgt_len]

        x = x.transpose(0, 1)   # src_len first
        encoder_output = encoder_output.transpose(0, 1)   # src_len first
        x += self.embed_positions(x, incremental=incremental)
        x = self.dropout(x)

        for layer in self.layers:
            x, attn_weights = layer(
                x,
                encoder_output,
                padding_mask=padding_mask,
                encoder_mask=encoder_mask,
                future_mask=future_mask,
                incremental=incremental,
            )
            if self.training:
                attn_weights = None

        x = self.layer_norm(x)
        x = x.transpose(0, 1)
        out = torch.matmul(x, self.embed_tokens.weight.T)
        return out, attn_weights


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, lr=1e-3, label_smoothing=0, use_cuda=True, max_len=50, clip=1.0,
                 scheduler=None, scheduler_args=None, amp=True):
        super().__init__()
        self.device = 'cuda' if (torch.cuda.is_available() and use_cuda) else 'cpu'
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)
        self.lr = lr
        self.max_len = max_len
        self.clip = clip
        self.amp = amp
        scheduler_args = scheduler_args or {}
        self.criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=PAD_IDX, label_smoothing=label_smoothing)

        if scheduler is None:
            self.scheduler_fn = ReduceLROnPlateau
            self.scheduler_args = dict(
                mode='max',
                patience=0,
                factor=0.1,      # when chrF plateaus, divide learning rate by 10
                threshold=0.5,   # reduce LR if chrF is lower than best chrF + 0.5 (i.e., if does not improve enough)
                threshold_mode='abs',
                verbose=True)
            dict.update(scheduler_args)   # can partially change the default values
        else:
            # can specify a different scheduler (e.g., ExponentialLR)
            # the scheduler's arguments need to be specified in full (as a dict)
            self.scheduler_fn = scheduler
            self.scheduler_args = scheduler_args
        
        self.optimizer = self.scheduler = self.scaler = None
        self.step_skipped = False
        self.epoch = 1

        self.START = torch.LongTensor([SOS_IDX]).to(self.device)
        self.metrics = {}

    @property
    def source_dict(self):
        return self.encoder.source_dict

    @property
    def target_dict(self):
        return self.decoder.target_dict

    def reset_optimizer(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.scheduler = self.scheduler_fn(self.optimizer, **self.scheduler_args)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        self.step_skipped = False
        self.epoch = 1

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

    def train_step(self, batch, train=True):
        input, input_len, target, target_len = batch['source'], batch['source_len'], batch['target'], batch['target_len']

        if input is None:
            return

        if train and self.optimizer is None:
            self.reset_optimizer()

        input = input.to(self.device)
        target = target.to(self.device)
        input_len = input_len.to(self.device)
        target_len = target_len.to(self.device)

        bsz = input.size(0)
        start = self.START.expand(bsz, 1)
        loss = 0
        if train:
            self.optimizer.zero_grad(set_to_none=True)
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()

        with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=(self.amp and train)):
            encoder_output = self.encoder(input, input_len=input_len)

        # Teacher forcing: Feed the target as the next input
        shifted_target = target.narrow(1, 0, target.size(1) - 1)
        decoder_input = torch.cat([start, shifted_target], 1)

        with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=(self.amp and train)):
            decoder_output, _ = self.decoder(
                input=decoder_input,
                input_len=input_len,
                encoder_output=encoder_output,
            )
        scores = decoder_output.view(-1, decoder_output.size(-1))
        loss = self.criterion(scores, target.view(-1)) / target_len.sum()
        
        if train:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            if self.clip is not None:
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip)
            self.scaler.step(self.optimizer)
            scale = self.scaler.get_scale()
            self.scaler.update()
            self.step_skipped = self.scaler.get_scale() != scale

        return loss.item()

    @torch.no_grad()
    def eval_step(self, batch):
        return self.train_step(batch, train=False)

    def scheduler_step(self, score=None, end_of_epoch=True):
        if end_of_epoch:
            self.epoch += 1
            
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(score)
            elif not isinstance(self.scheduler, WarmupLR):
                if not self.step_skipped:
                    self.scheduler.step()
        
        elif isinstance(self.scheduler, WarmupLR):
            if not self.step_skipped:
                self.scheduler.step()

    @torch.no_grad()
    def translate(self, batch):
        input, input_len = batch['source'], batch['source_len']

        if input is None:
            return

        input = input.to(self.device)
        input_len = input_len.to(self.device)

        bsz = input.size(0)
        bos = self.START.expand(bsz, 1)  # expand to batch size
        prefix = batch.get('prefix')

        if prefix is None:
            prefix = bos
        else:
            prefix = prefix.to(self.device)
            prefix = torch.cat([bos, prefix], dim=1)

        self.encoder.eval()
        self.decoder.eval()
        # not doing autocast here: for some reason here AMP is slower than full precision here
        encoder_output = self.encoder(input, input_len=input_len)

        predictions = []
        done = [False for _ in range(bsz)]
        total_done = 0
        decoder_input = prefix
        attn_weights = []

        for _ in range(self.max_len):
            # generate at most max_len tokens
            decoder_output, attn_weights_ = self.decoder(
                input=decoder_input,
                input_len=input_len,
                encoder_output=encoder_output,
                incremental=True,
            )

            _, preds = decoder_output.max(dim=2)
            preds = preds[:,-1:]
            predictions.append(preds.cpu())
            decoder_input = preds  # set input to next step

            attn_weights.append(attn_weights_)

            # check if we've produced the end token
            for b in range(bsz):
                if not done[b]:
                    # only add more tokens for examples that aren't done
                    if preds[b][-1] == EOS_IDX:
                        # if we produced END, we're done
                        done[b] = True
                        total_done += 1
            if total_done == bsz:
                # no need to generate any more
                break
        
        self.decoder.reset_state()
        
        predictions = torch.cat(predictions, 1)
        
        attn = None if attn_weights[0] is None else torch.cat(attn_weights, 1).float().detach().cpu().numpy()
        return self.vec2txt(predictions), attn

    def load(self, path, strict=True, reset_optimizer=False):
        if not os.path.isfile(path):
            return
        
        ckpt = torch.load(path, map_location='cpu')
        model = ckpt['model']

        # Parameters that are sometimes present in fairseq checkpoints and that we don't need:
        model.pop('encoder.embed_positions._float_tensor', None)
        model.pop('decoder.embed_positions._float_tensor', None)
        model.pop('encoder.version', None)
        model.pop('decoder.version', None)

        if 'decoder.embed_tokens.weight' not in model:
            model['decoder.embed_tokens.weight'] = model['encoder.embed_tokens.weight']
        
        for k, v in self.state_dict().items():
            # automatically add random adapter parameters to the model checkpoint to
            # avoid any error when initializing adapter models
            if k not in model and 'adapters' in k.split('.'):
                model[k] = v

        self.load_state_dict(model, strict=strict)
        
        if reset_optimizer and self.optimizer is not None:
            self.reset_optimizer()
        elif not reset_optimizer:
            self.reset_optimizer()
            if ckpt.get('scheduler'):
                state_dict = ckpt['scheduler']
                state_dict.pop('best', None)  # scores won't be comparable if we change valid set
                self.scheduler.load_state_dict(state_dict)
            if ckpt.get('optimizer'):
                self.optimizer.load_state_dict(ckpt['optimizer'])
            if ckpt.get('scaler'):
                self.scaler.load_state_dict(ckpt['scaler'])
            if ckpt.get('metrics'):
                self.metrics = ckpt['metrics']
            if ckpt.get('epoch'):
                self.epoch = ckpt['epoch']

    def save(self, path):
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        ckpt = {
            'model': self.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'metrics': self.metrics,
            'epoch': self.epoch,
            'scaler': self.scaler.state_dict(),
        }
        torch.save(ckpt, path)

    def record(self, name, value):
        self.metrics.setdefault(self.epoch, {})[name] = value

    @contextmanager
    def adapter(self, id, projection_dim=64, overwrite=False):
        """
        Temporarily activates adapters, for example:
        
        with model.adapter('en-fr'):
            translate(model, 'Hello, world')
        """
        self.encoder.add_adapter(id, projection_dim, select=True, overwrite=overwrite)
        self.decoder.add_adapter(id, projection_dim, select=True, overwrite=overwrite)
        yield
        self.encoder.select_adapter(None)
        self.decoder.select_adapter(None)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0, ffn_dim=None, activation=F.relu):
        super().__init__()
        self.embed_dim = embed_dim
        self.self_attn = MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.fc1 = nn.Linear(embed_dim, ffn_dim or embed_dim)
        self.fc2 = nn.Linear(ffn_dim or embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.activation = activation
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        x = src
        residual = x
        x = self.self_attn_layer_norm(x)
        x, _, _ = self.self_attn(
            x, x, x, attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask)
        x = residual + self.dropout(x)
        residual = x
        x = self.final_layer_norm(x)
        x = self.fc2(self.dropout(self.activation(self.fc1(x))))
        x = residual + self.dropout(x)
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0, ffn_dim=None, activation=F.relu):
        super().__init__()
        self.embed_dim = embed_dim
        self.self_attn = MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.encoder_attn = MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.fc1 = nn.Linear(embed_dim, ffn_dim or embed_dim)
        self.fc2 = nn.Linear(ffn_dim or embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.encoder_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.activation = activation

    def forward(self, tgt, memory, future_mask=None, memory_mask=None,
                padding_mask=None, encoder_mask=None, incremental=False):
        x = tgt
        residual = x
        x = self.self_attn_layer_norm(x)
        x, _, _ = self.self_attn(
            x, x, x, attn_mask=future_mask,
            key_padding_mask=padding_mask, incremental=incremental)
        x = residual + self.dropout(x)

        residual = x
        x = self.encoder_attn_layer_norm(x)
        x, attn_weights, _ = self.encoder_attn(
            x, memory, memory, attn_mask=memory_mask,
            key_padding_mask=encoder_mask)
        x = residual + self.dropout(x)

        residual = x
        x = self.final_layer_norm(x)
        x = self.fc2(self.dropout(self.activation(self.fc1(x))))
        x = residual + self.dropout(x)

        return x, attn_weights


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=128):
        super(PositionalEncoding, self).__init__()
        max_len += PAD_IDX + 1   # like in fairseq
        half_dim = embed_dim // 2
        weight = math.log(10000) / (half_dim - 1)
        weight = torch.exp(torch.arange(half_dim, dtype=torch.float) * -weight)
        weight = torch.arange(max_len, dtype=torch.float).unsqueeze(1) * weight.unsqueeze(0)
        weight = torch.cat([torch.sin(weight), torch.cos(weight)], dim=1).view(max_len, -1)
        weight[PAD_IDX, :] = 0
        self.weight = weight.unsqueeze(0).transpose(0, 1)
        self.reset_state()

    def reset_state(self):
        self.state = PAD_IDX + 1

    def forward(self, x, incremental=False):
        length = x.size(0)
        x = self.weight[self.state:self.state + length, :].to(x.device)
        if incremental:
            self.state += length
        return x


class AdapterLayer(nn.Module):
    def __init__(self, input_dim, projection_dim):
        super().__init__()
        self.down = nn.Linear(input_dim, projection_dim)
        self.up = nn.Linear(projection_dim, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)
        nn.init.uniform_(self.down.weight, -1e-6, 1e-6)
        nn.init.uniform_(self.up.weight, -1e-6, 1e-6)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        y = self.layer_norm(x)
        y = self.down(y)
        y = F.relu(y)
        y = self.up(y)
        return x + y


class AdapterTransformerEncoderLayer(TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adapters = nn.ModuleDict({})
        self.adapter_id = None

    def add_adapter(self, id, projection_dim, overwrite=False):
        if overwrite or id not in self.adapters:
            device = next(iter(self.parameters())).device
            adapter = AdapterLayer(self.embed_dim, projection_dim).to(device)
            self.adapters[id] = adapter

    def forward(self, *args, **kwargs):
        x = super().forward(*args, **kwargs)
        if self.adapter_id is not None:
            x = self.adapters[self.adapter_id](x)
        return x


class AdapterTransformerEncoder(TransformerEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for param in self.parameters():
            param.requires_grad = False

    def add_adapter(self, id, projection_dim, select=False, overwrite=False):
        for layer in self.layers:
            layer.add_adapter(id, projection_dim, overwrite=overwrite)
        if select:
            self.select_adapter(id)

    def select_adapter(self, id):
        for layer in self.layers:
            assert id is None or id in layer.adapters
            layer.adapter_id = id

    def build_layer(self, layer_id):
        return AdapterTransformerEncoderLayer(self.embed_dim, self.heads, self.dropout_rate, self.ffn_dim)


class AdapterTransformerDecoderLayer(TransformerDecoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adapters = nn.ModuleDict({})
        self.adapter_id = None
    
    def add_adapter(self, id, projection_dim, overwrite=False):
        if overwrite or id not in self.adapters:
            device = next(iter(self.parameters())).device
            adapter = AdapterLayer(self.embed_dim, projection_dim).to(device)
            self.adapters[id] = adapter

    def forward(self, *args, **kwargs):
        x, attn = super().forward(*args, **kwargs)
        if self.adapter_id is not None:
            x = self.adapters[self.adapter_id](x)
        return x, attn


class AdapterTransformerDecoder(TransformerDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for param in self.parameters():
            param.requires_grad = False

    def add_adapter(self, id, projection_dim, select=False, overwrite=False):
        for layer in self.layers:
            layer.add_adapter(id, projection_dim, overwrite=overwrite)
        if select:
            self.select_adapter(id)

    def select_adapter(self, id):
        for layer in self.layers:
            assert id is None or id in layer.adapters
            layer.adapter_id = id

    def build_layer(self, layer_id):
        return AdapterTransformerDecoderLayer(self.embed_dim, self.heads, self.dropout_rate, self.ffn_dim)


class WarmupLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, last_epoch=-1, warmup=1000, init_lr=0.0, verbose=False):
        self.warmup = warmup
        self.init_lr = init_lr
        param_group = next(iter(optimizer.param_groups))
        self.lr = param_group.get('initial_lr', param_group['lr'])
        if self.init_lr < 0:
            self.init_lr = 0 if self.warmup > 0 else self.lr
        self.lr_step = (self.lr - self.init_lr) / self.warmup
        self.decay_factor = self.lr * self.warmup ** 0.5
        super(WarmupLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch < self.warmup:
            lr = self.init_lr + self.last_epoch * self.lr_step
        else:
            lr = self.decay_factor * self.last_epoch ** -0.5
        return [lr] * len(self.optimizer.param_groups)
