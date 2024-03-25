import os
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, LRScheduler
from torch import Tensor, BoolTensor, LongTensor
import math
import functools
from typing import Optional, Union, Any
from data import Dictionary
from contextlib import contextmanager


def checkpoint(module: nn.Module):
    module._orig_forward = module.forward
    from torch.utils.checkpoint import checkpoint
    module.forward = functools.partial(checkpoint, module._orig_forward, use_reentrant=False)
    return module


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt((x**2).mean(-1, keepdim=True) + self.eps) * self.weight
        return x.to(dtype)


class Encoder(nn.Module):
    def __init__(
        self,
        source_dict: Dictionary,
        embed_dim: int = 512,
        num_layers: int = 1,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.source_dict = source_dict
        self.pad_idx = source_dict.pad_idx
        self.eos_idx = source_dict.eos_idx
        self.sos_idx = source_dict.sos_idx
        self.input_size = len(source_dict)
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.embed_tokens = nn.Embedding(self.input_size, self.embed_dim, padding_idx=self.pad_idx)
        nn.init.normal_(self.embed_tokens.weight, mean=0, std=self.embed_tokens.weight.size(-1) ** -0.5)
        nn.init.constant_(self.embed_tokens.weight[self.pad_idx], 0)
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def select_adapter(self, id: str) -> None:
        pass


class Decoder(nn.Module):
    def __init__(
        self,
        target_dict: Dictionary,
        embed_dim: int = 512,
        num_layers: int = 1,
        dropout: float = 0.0,
        embed_tokens: Optional[nn.Embedding] = None,
        **kwargs,
    ):
        super().__init__()
        self.target_dict = target_dict
        self.pad_idx = target_dict.pad_idx
        self.eos_idx = target_dict.eos_idx
        self.sos_idx = target_dict.sos_idx
        self.output_size = len(target_dict)
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        if embed_tokens is None:
            self.embed_tokens = nn.Embedding(self.output_size, self.embed_dim, padding_idx=self.pad_idx)
            nn.init.normal_(self.embed_tokens.weight, mean=0, std=self.embed_tokens.weight.size(-1) ** -0.5)
            nn.init.constant_(self.embed_tokens.weight[self.pad_idx], 0)
        else:
            self.embed_tokens = embed_tokens
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def select_adapter(self, id: str) -> None:
        pass



class MultiheadAttention(nn.Module):  # copied from Pasero
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, rope: bool = False, causal: bool = False,
                 has_bias: bool = True, kv_heads: Optional[int] = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        kv_heads = kv_heads or num_heads  # for grouped-query attention
        self.kv_heads = kv_heads
        self.dropout = dropout

        self.head_dim = embed_dim // num_heads  # does not work with T5 small
        self.kv_dim = self.kv_heads * self.head_dim
        self.q_dim = self.num_heads * self.head_dim
        assert num_heads % kv_heads == 0
        self.has_bias = has_bias
        
        self.k_proj = nn.Linear(embed_dim, self.kv_dim, bias=self.has_bias)
        self.v_proj = nn.Linear(embed_dim, self.kv_dim, bias=self.has_bias)
        self.q_proj = nn.Linear(embed_dim, self.q_dim, bias=self.has_bias)
        self.out_proj = nn.Linear(self.q_dim, embed_dim, bias=self.has_bias)
        self.rotary_embed = RotaryEmbedding(self.head_dim) if rope else None
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        self.causal = causal
        self.causal_mask = torch.empty(0, dtype=torch.bool)
        self.reset_state()

    def reset_state(self) -> None:
        self.state = {}
    
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        padding_mask: Optional[BoolTensor] = None,
        return_attn: bool = False,
        incremental: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """
        Shape:
            query: (B, T, D)
            key:   (B, S, D)
            value: (B, S, D)
            padding_mask: (B, S)
        
        Returns: tuple (attn, attn_weights) with
            attn: tensor of shape (B, T, D)
            attn_weights: tensor of shape (B, T, H, S) or None
        """
        if padding_mask is not None and self.causal:
            # attn_mask can be a simple padding mask, in which case it is useless for causal attention (assuming the
            # padding tokens are always at the end...)
            padding_mask = None  # set to None to allow the use of fast causal attention below
        
        batch_size, tgt_len, embed_dim = query.size()
        src_len = key.size(1)

        q = self.q_proj(query)  # BxTxD
        q = q.view(batch_size, tgt_len, self.num_heads, self.head_dim)  # BxTxHxD'
        k = self.k_proj(key)  # BxSxD
        k = k.view(batch_size, -1, self.kv_heads, self.head_dim)  # BxSxH'xD'
        v = self.v_proj(value)
        v = v.view(batch_size, -1, self.kv_heads, self.head_dim)  # BxSxH'xD'

        pos_offset = self.state['key'].size(1) if self.state and incremental else 0
        
        if self.rotary_embed is not None:
            q, k = self.rotary_embed(q, k, offset=pos_offset)

        if incremental:
            if self.state:  # step > 0
                prev_key = self.state['key']  # BxSxHxD'
                prev_value = self.state['value']  # BxSxHxD'
                k = torch.cat([prev_key, k], dim=1)
                v = torch.cat([prev_value, v], dim=1)
                src_len = k.size(1)

            self.state['key'] = k
            self.state['value'] = v

        r = self.num_heads // self.kv_heads
        if r > 1:
            v = v.repeat_interleave(r, dim=2)
            k = k.repeat_interleave(r, dim=2)

        if self.head_dim > 64:
            return_attn = True  # dirty bug fix with flash attention

        attn_mask = None
        if return_attn or padding_mask is not None:  # custom masking
            if padding_mask is not None:
                attn_mask = padding_mask[:,None,None,:]  # Bx1x1xS

            if self.causal:
                if self.causal_mask.size(0) < src_len:
                    size = 256 * math.ceil(src_len / 256)
                    self.causal_mask = torch.ones(size, size, dtype=torch.bool, device=q.device)
                    self.causal_mask = torch.triu(self.causal_mask, 1)
                    
                self.causal_mask = self.causal_mask.to(q.device)
                causal_mask = self.causal_mask[:src_len, :src_len]
                causal_mask = causal_mask[-tgt_len:]
                causal_mask = causal_mask.view(1, 1, tgt_len, src_len)  # 1x1xTxS
                attn_mask = causal_mask if attn_mask is None else (attn_mask + causal_mask)
            
            if attn_mask is not None:
                attn_mask = attn_mask.to(q.dtype).masked_fill(attn_mask, -float('inf'))  # bool -> float

        dropout_p = self.dropout if self.training else 0
        is_causal = self.causal and tgt_len > 1 and attn_mask is None  # let Pytorch compute the causal mask

        if not return_attn:
            q = q.transpose(1, 2)  # BxHxTxD'
            k = k.transpose(1, 2)  # BxHxSxD'
            v = v.transpose(1, 2)  # BxHxSxD'
            attn: Tensor = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                is_causal=is_causal,
                dropout_p=dropout_p,
            )  # BxHxTxD'
            attn_weights = None
            attn = attn.transpose(1, 2)
        else:
            # use custom attention if we need the attention weights or flash attention is not available (e.g., 
            # Pytorch version that is too old)
            q = q.transpose(1, 2)  # BxHxTxD'
            k = k.transpose(1, 2)  # BxHxSxD'
            v = v.transpose(1, 2)  # BxHxSxD'
            attn, attn_weights = self.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
            )  # BxHxTxD'
            attn = attn.transpose(1, 2)
        
        attn = attn.reshape(batch_size, tgt_len, -1)  # BxTxD
        attn = self.out_proj(attn)

        return attn, attn_weights

    @classmethod
    def scaled_dot_product_attention(
        cls,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Tensor,
        dropout_p: float = 0.0,
    ) -> tuple[Tensor, Tensor]:
        # q: BxHxTxD
        # k: BxHxSxD
        # v: BxHxSxD
        # attn_mask: BxHxTxS
        head_dim = q.shape[-1]
        scale = 1.0 / head_dim ** 0.5

        attn_weights = torch.matmul(q, k.transpose(2, 3)) * scale  # BxHxTxS

        if attn_mask is not None:
            attn_weights += attn_mask
        
        attn_weights_float = F.softmax(attn_weights, dim=-1, dtype=torch.float)
        attn_weights_float = attn_weights_float.nan_to_num()  # NaNs can happen with BLOOM models where the beginning
        # of sentence is replaced by a padding token
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_weights_ = F.dropout(attn_weights, p=dropout_p)  # BxHxTxS
        attn = torch.matmul(attn_weights_, v)  # BxHxTxD
        attn_weights = attn_weights.transpose(1, 2)  # BxTxHxS
        return attn, attn_weights


class RotaryEmbedding(nn.Module):  # copied from Pasero
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.build(max_len=256)  # will be automatically extended if needed

    def build(self, max_len: int):
        t = torch.arange(max_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos = emb.cos()
        self.sin = emb.sin()
        self.max_len = max_len

    def rotate(self, x):
        x1 = x[..., :self.dim // 2]
        x2 = x[..., self.dim // 2:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, query: Tensor, key: Tensor, offset: Union[LongTensor, int]) -> tuple[Tensor, Tensor]:
        """
        :param query: tensor of shape (B, T, H, D)
        :param key: tensor of shape (B, S, H, D)
        :param offset: number of previous tokens (during incremental decoding), can be a tensor of shape (B,) if the
            length of the each sequence in the batch is different (due to different padding)
        
        Returns: rotated query and key
        """
        bsz, seq_len, _, dim = query.shape

        total_len = offset.max() if torch.is_tensor(offset) else offset
        total_len += seq_len
        if total_len > self.max_len:  # extend the size of the embeddings if needed
            new_max_len = 2**math.ceil(math.log2(total_len))  # closest power of 2
            self.build(new_max_len)

        self.cos = self.cos.to(query)  # device and dtype
        self.sin = self.sin.to(query)

        if torch.is_tensor(offset) and offset.dim() == 1:
            cos = self.cos.repeat(bsz, 1, 1)  # B x MAX_LEN x D
            sin = self.sin.repeat(bsz, 1, 1)  # B x MAX_LEN x D
            positions = torch.arange(seq_len, device=query.device).unsqueeze(0) + offset.unsqueeze(1)  # BxT
            positions = positions.unsqueeze(-1).repeat(1, 1, cos.size(-1))  # BxTxD
            cos = torch.gather(cos, dim=1, index=positions)  # BxTxD
            sin = torch.gather(sin, dim=1, index=positions)  # BxTxD
            cos = cos.unsqueeze(2)  # BxTx1xD
            sin = sin.unsqueeze(2)  # BxTx1xD
        else:
            cos = self.cos[None, offset : offset + seq_len, None]  # 1xTx1xD
            sin = self.sin[None, offset : offset + seq_len, None]  # 1xTx1xD

        q = query[..., :self.dim]  # BxTxHxD
        k = key[..., :self.dim]
    
        q = (q * cos) + (self.rotate(q) * sin)
        k = (k * cos) + (self.rotate(k) * sin)

        if self.dim < dim:
            q = torch.cat([q, query[..., self.dim:]], dim=-1)
            k = torch.cat([k, key[..., self.dim:]], dim=-1)
        return q, k


class BOW_Encoder(Encoder):
    def __init__(self, *args, reduce: str = "max", **kwargs):
        super().__init__(*args, **kwargs)
        self.reduce = reduce
        assert(self.reduce in ["sum", "mean", "max"])
        self.activation = nn.ReLU()
        self.layers = nn.ModuleList([nn.Linear(self.embed_dim, self.embed_dim)])
        for _ in range(self.num_layers - 1):
            self.layers.append(nn.Linear(self.embed_dim, self.embed_dim))

    def forward(self, input: LongTensor, input_len: LongTensor, **kwargs):
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

    def forward(self, input: LongTensor, input_len: LongTensor, **kwargs):
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

    def reset_state(self) -> None:
        self.state = None

    def forward(
        self,
        input: LongTensor,
        input_len: LongTensor,
        encoder_output: Tensor,
        incremental: bool = False,
        **kwargs,
    ) -> tuple[Tensor, None]:
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
    def __init__(
        self,
        *args,
        num_heads: int = 4,
        ffn_dim: Optional[int] = None,
        checkpointing: bool = False,
        has_bias: bool = True,
        has_encoder: bool = True,
        activation: str = 'relu',
        rms_norm: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        LayerNorm = RMSNorm if rms_norm else nn.LayerNorm
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim or self.embed_dim
        self.embed_positions = PositionalEncoding(self.embed_dim, self.pad_idx)
        self.has_encoder = has_encoder
        self.has_bias = has_bias
        self.activation = activation
        self.rms_norm = rms_norm
        self.layers = nn.ModuleList([self.build_layer(i) for i in range(self.num_layers)])
        if checkpointing:
            for layer in self.layers:
                checkpoint(layer)
        self.layer_norm = LayerNorm(self.embed_dim)
        self.embed_scale = math.sqrt(self.embed_dim)
        self.dropout = nn.Dropout(self.dropout_rate)
    
    def build_layer(self, layer_id: int) -> 'TransformerEncoderLayer':
        return TransformerEncoderLayer(
            self.embed_dim,
            self.num_heads,
            self.dropout_rate,
            self.ffn_dim,
            has_bias=self.has_bias,
            activation=self.activation,
            rms_norm=self.rms_norm,
        )

    def forward(self, input: LongTensor, input_len: int, **kwargs) -> Tensor:
        x = self.embed_tokens(input) * self.embed_scale

        mask = torch.arange(input.size(1))[None, :].to(input_len.device) >= input_len[:, None]
        # shape: (batch_size, src_len)

        x += self.embed_positions(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, padding_mask=mask)

        return self.layer_norm(x)


class TransformerDecoder(Decoder):
    def __init__(
        self, *args,
        num_heads: int = 4,
        ffn_dim: Optional[int] = None,
        checkpointing: bool = False,
        has_bias: bool = True,
        has_encoder: bool = True,
        activation: str = 'relu',
        rms_norm: bool = False,
        tied_embed: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        LayerNorm = RMSNorm if rms_norm else nn.LayerNorm
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim or self.embed_dim
        self.embed_positions = PositionalEncoding(self.embed_dim, self.pad_idx)  # TODO (Llama)
        self.has_encoder = has_encoder
        self.has_bias = has_bias
        self.activation = activation
        self.rms_norm = rms_norm
        self.layers = nn.ModuleList([self.build_layer(i) for i in range(self.num_layers)])
        if checkpointing:
            for layer in self.layers:
                checkpoint(layer)
        self.layer_norm = LayerNorm(self.embed_dim)
        self.output_projection = (
            None if tied_embed else
            nn.Linear(self.embed_dim, self.embed_tokens.num_embeddings, has_bias=False)
        )
        self.embed_scale = math.sqrt(self.embed_dim)
        self.dropout = nn.Dropout(self.dropout_rate)

    def build_layer(self, layer_id: int) -> 'TransformerDecoderLayer':
        return TransformerDecoderLayer(
            self.embed_dim,
            self.num_heads,
            self.dropout_rate,
            self.ffn_dim,
            has_bias=self.has_bias,
            has_encoder=self.has_encoder,
            activation=self.activation,
            rms_norm=self.rms_norm,
        )

    def reset_state(self) -> None:
        self.embed_positions.reset_state()
        for layer in self.layers:
            layer.self_attn.reset_state()

    def forward(
        self,
        input: LongTensor,
        input_len: LongTensor,
        encoder_output: Tensor,
        incremental: bool = False,
        return_attn: bool = False,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        """
        input: Tensor of shape (batch_size, tgt_len)
        input_len: Tensor of shape (batch_size,)
        encoder_output: Tensor of shape (batch_size, src_len, embed_dim)
        """
        x = self.embed_tokens(input) * self.embed_scale
        src_len = encoder_output.size(1)
        encoder_mask = torch.arange(src_len)[None, :].to(input_len.device) >= input_len[:, None]

        x += self.embed_positions(x, incremental=incremental)
        x = self.dropout(x)

        for layer in self.layers:
            x, attn_weights = layer(
                x,
                encoder_output,
                encoder_mask=encoder_mask,
                incremental=incremental,
                return_attn=return_attn,
            )
            if self.training:
                attn_weights = None

        x = self.layer_norm(x)
        if self.output_projection is None:
            out = torch.matmul(x, self.embed_tokens.weight.T)
        else:
            out = self.output_projection(x)
        
        return out, attn_weights


class EncoderDecoder(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        lr: float = 1e-3,
        label_smoothing: float = 0.0,
        use_cuda: bool = True,
        max_len: int = 50,
        clip: float = 1.0,
        scheduler: Optional[LRScheduler] = None,
        scheduler_args: Optional[dict] = None,
        amp: bool = True,
    ):
        super().__init__()
        self.device = 'cuda' if (torch.cuda.is_available() and use_cuda) else 'cpu'
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)
        self.lr = lr
        self.max_len = max_len
        self.clip = clip
        self.amp = amp  # speeds up training on recent GPU by training in float16
        scheduler_args = scheduler_args or {}
        self.criterion = nn.CrossEntropyLoss(
            reduction='sum',
            ignore_index=decoder.pad_idx,
            label_smoothing=label_smoothing,
        )

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

        self.START = torch.LongTensor([decoder.sos_idx]).to(self.device)
        self.metrics = {}

    @property
    def source_dict(self) -> Dictionary:
        return self.encoder.source_dict

    @property
    def target_dict(self) -> Dictionary:
        return self.decoder.target_dict

    def reset_optimizer(self) -> None:
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.scheduler = self.scheduler_fn(self.optimizer, **self.scheduler_args)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)  # to avoid overflows with float16 training
        self.step_skipped = False
        self.epoch = 1

    def vec2txt(self, vector: LongTensor) -> tuple[str, list[str]]:
        """
        Convert vector to text.
        :param vector: tensor of token indices.
        1-d tensors will return a string, 2-d will return a list of strings
        """
        if vector.dim() == 1:
            output_tokens = []
            # Remove the final END_TOKEN that is appended to predictions
            for token in vector:
                if token == self.decoder.eos_idx:
                    break
                else:
                    output_tokens.append(token)
            return self.target_dict.vec2txt(output_tokens)

        elif vector.dim() == 2:
            return [self.vec2txt(vector[i]) for i in range(vector.size(0))]
        raise RuntimeError(
            "Improper input to vec2txt with dimensions {}".format(vector.size())
        )

    def train_step(self, batch: dict[str, Any], train: bool = True) -> float:
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
    def eval_step(self, batch: dict[str, Any]) -> float:
        return self.train_step(batch, train=False)

    def scheduler_step(self, score: Optional[float] = None, end_of_epoch: bool = True) -> None:
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
    def translate(self, batch: dict[str, Any], return_attn: bool = False) -> tuple[list[str], Optional[Tensor]]:
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
                return_attn=return_attn,
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
                    if preds[b][-1] == self.decoder.eos_idx:
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

    def load(self, path: str, strict: bool = True, reset_optimizer: bool = False) -> None:
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

    def save(self, path: str) -> None:
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

    def record(self, name: str, value: float) -> None:
        self.metrics.setdefault(self.epoch, {})[name] = value

    @contextmanager
    def adapter(self, id: str, projection_dim: int = 64, overwrite: bool = False):
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
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        ffn_dim: Optional[int] = None,
        has_bias: bool = True,
        activation: str = 'relu',
        rms_norm: bool = False,
    ):
        super().__init__()
        LayerNorm = RMSNorm if rms_norm else nn.LayerNorm
        self.embed_dim = embed_dim
        self.self_attn = MultiheadAttention(embed_dim, num_heads, dropout=dropout, has_bias=has_bias)
        self.fc1 = nn.Linear(embed_dim, ffn_dim or embed_dim)
        self.fc2 = nn.Linear(ffn_dim or embed_dim, embed_dim)
        self.fc3 = nn.Linear(embed_dim, ffn_dim, bias=has_bias) if activation == 'swiglu' else None
        self.dropout = nn.Dropout(dropout)
        self.self_attn_layer_norm = LayerNorm(embed_dim)
        self.final_layer_norm = LayerNorm(embed_dim)
        self.activation_fn = F.silu if activation == 'swiglu' else getattr(F, activation)
    
    def forward(self, src: Tensor, padding_mask: Optional[BoolTensor] = None) -> Tensor:
        x = src
        residual = x
        x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(x, x, x, padding_mask=padding_mask)
        x = residual + self.dropout(x)
        residual = x
        x = self.final_layer_norm(x)
        y = self.fc1(x)
        y = self.activation_fn(y)
        y = self.dropout(y)
        if self.fc3 is not None:
            y = y * self.fc3(x)
        x = self.fc2(y)
        x = residual + self.dropout(x)
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        ffn_dim: Optional[int] = None,
        has_bias: bool = True,
        activation: str = 'relu',
        has_encoder: bool = True,
        rms_norm: bool = False,
    ):
        super().__init__()
        LayerNorm = RMSNorm if rms_norm else nn.LayerNorm
        self.embed_dim = embed_dim
        self.self_attn = MultiheadAttention(embed_dim, num_heads, dropout=dropout, causal=True, has_bias=has_bias)
        self.has_encoder = has_encoder
        if has_encoder:
            self.encoder_attn = MultiheadAttention(embed_dim, num_heads, dropout=dropout, has_bias=has_bias)
        self.fc1 = nn.Linear(embed_dim, ffn_dim or embed_dim, bias=has_bias)
        self.fc2 = nn.Linear(ffn_dim or embed_dim, embed_dim, bias=has_bias)
        self.fc3 = nn.Linear(embed_dim, ffn_dim, bias=has_bias) if activation == 'swiglu' else None
        self.dropout = nn.Dropout(dropout)
        self.self_attn_layer_norm = LayerNorm(embed_dim)
        if has_encoder:
            self.encoder_attn_layer_norm = LayerNorm(embed_dim)
        self.final_layer_norm = LayerNorm(embed_dim)
        self.activation_fn = F.silu if activation == 'swiglu' else getattr(F, activation)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        encoder_mask: Optional[BoolTensor] = None,
        incremental: bool = False,
        return_attn: bool = False
    ) -> tuple[Tensor, Optional[Tensor]]:
        x = tgt
        residual = x
        x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            x, x, x, incremental=incremental, return_attn=False)
        x = residual + self.dropout(x)

        if self.has_encoder:
            residual = x
            x = self.encoder_attn_layer_norm(x)
            x, attn_weights = self.encoder_attn(
                x, memory, memory, padding_mask=encoder_mask, return_attn=return_attn)
            x = residual + self.dropout(x)
            if attn_weights is not None:  # BxTxHxS
                attn_weights = attn_weights.sum(dim=2) / attn_weights.size(2)  # average all attention heads

        residual = x
        x = self.final_layer_norm(x)
        y = self.fc1(x)
        y = self.activation_fn(y)
        y = self.dropout(y)
        if self.fc3 is not None:
            y = y * self.fc3(x)
        x = self.fc2(y)
        x = residual + self.dropout(x)

        return x, attn_weights


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, pad_idx: int, max_len: int = 128, shift: int = 2):
        super(PositionalEncoding, self).__init__()
        self.pad_idx = pad_idx
        self.shift = shift
        max_len += shift  # like in fairseq
        half_dim = embed_dim // 2
        weight = math.log(10000) / (half_dim - 1)
        weight = torch.exp(torch.arange(half_dim, dtype=torch.float) * -weight)
        weight = torch.arange(max_len, dtype=torch.float).unsqueeze(1) * weight.unsqueeze(0)
        weight = torch.cat([torch.sin(weight), torch.cos(weight)], dim=1).view(max_len, -1)
        weight[pad_idx, :] = 0
        self.weight = weight.unsqueeze(0)
        self.reset_state()

    def reset_state(self) -> None:
        self.state = self.shift

    def forward(self, x: LongTensor, incremental: bool = False) -> Tensor:
        length = x.size(1)
        x = self.weight[:, self.state:self.state + length].to(x.device)
        if incremental:
            self.state += length
        return x


class AdapterLayer(nn.Module):
    def __init__(self, input_dim: int, projection_dim: int):
        super().__init__()
        self.down = nn.Linear(input_dim, projection_dim)
        self.up = nn.Linear(projection_dim, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)
        nn.init.uniform_(self.down.weight, -1e-6, 1e-6)
        nn.init.uniform_(self.up.weight, -1e-6, 1e-6)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: Tensor) -> Tensor:
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

    def add_adapter(self, id: str, projection_dim: int, overwrite: bool = False) -> None:
        if overwrite or id not in self.adapters:
            device = next(iter(self.parameters())).device
            adapter = AdapterLayer(self.embed_dim, projection_dim).to(device)
            self.adapters[id] = adapter

    def forward(self, *args, **kwargs) -> Tensor:
        x = super().forward(*args, **kwargs)
        if self.adapter_id is not None:
            x = self.adapters[self.adapter_id](x)
        return x


class AdapterTransformerEncoder(TransformerEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for param in self.parameters():
            param.requires_grad = False

    def add_adapter(self, id: str, projection_dim: int, select: bool = False, overwrite: bool = False) -> None:
        for layer in self.layers:
            layer.add_adapter(id, projection_dim, overwrite=overwrite)
        if select:
            self.select_adapter(id)

    def select_adapter(self, id: str) -> None:
        for layer in self.layers:
            assert id is None or id in layer.adapters
            layer.adapter_id = id

    def build_layer(self, layer_id: int) -> 'AdapterTransformerEncoderLayer':
        return AdapterTransformerEncoderLayer(self.embed_dim, self.num_heads, self.dropout_rate, self.ffn_dim)


class AdapterTransformerDecoderLayer(TransformerDecoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adapters = nn.ModuleDict({})
        self.adapter_id = None
    
    def add_adapter(self, id: str, projection_dim: int, overwrite: bool = False) -> None:
        if overwrite or id not in self.adapters:
            device = next(iter(self.parameters())).device
            adapter = AdapterLayer(self.embed_dim, projection_dim).to(device)
            self.adapters[id] = adapter

    def forward(self, *args, **kwargs) -> tuple[Tensor, Optional[Tensor]]:
        x, attn = super().forward(*args, **kwargs)
        if self.adapter_id is not None:
            x = self.adapters[self.adapter_id](x)
        return x, attn


class AdapterTransformerDecoder(TransformerDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for param in self.parameters():
            param.requires_grad = False

    def add_adapter(self, id: str, projection_dim: int, select: bool = False, overwrite: bool = False) -> None:
        for layer in self.layers:
            layer.add_adapter(id, projection_dim, overwrite=overwrite)
        if select:
            self.select_adapter(id)

    def select_adapter(self, id: str) -> None:
        for layer in self.layers:
            assert id is None or id in layer.adapters
            layer.adapter_id = id

    def build_layer(self, layer_id: int) -> 'AdapterTransformerDecoderLayer':
        return AdapterTransformerDecoderLayer(self.embed_dim, self.num_heads, self.dropout_rate, self.ffn_dim)


class WarmupLR(LRScheduler):
    def __init__(
        self,
        optimizer: optim.Optimizer,
        last_epoch: int = -1,
        warmup: int = 1000,
        init_lr: float = 0.0,
        verbose: bool = False,
    ):
        self.warmup = warmup
        self.init_lr = init_lr
        param_group = next(iter(optimizer.param_groups))
        self.lr = param_group.get('initial_lr', param_group['lr'])
        if self.init_lr < 0:
            self.init_lr = 0 if self.warmup > 0 else self.lr
        self.lr_step = (self.lr - self.init_lr) / self.warmup
        self.decay_factor = self.lr * self.warmup ** 0.5
        super(WarmupLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> list[float]:
        if self.last_epoch < self.warmup:
            lr = self.init_lr + self.last_epoch * self.lr_step
        else:
            lr = self.decay_factor * self.last_epoch ** -0.5
        return [lr] * len(self.optimizer.param_groups)
