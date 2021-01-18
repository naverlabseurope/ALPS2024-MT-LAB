import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
# for monkey patching the default multi_head_attention_forward method
from torch.nn.functional import *
import math
from collections import namedtuple

import sacrebleu

from nmt_dataset import PAD_IDX, SOS_IDX, EOS_IDX


def alps_multi_head_attention_forward(
    query,
    key,
    value,
    embed_dim_to_check,
    num_heads,
    in_proj_weight,
    in_proj_bias,
    bias_k,
    bias_v,
    add_zero_attn,
    dropout_p,
    out_proj_weight,
    out_proj_bias,
    training=True,
    key_padding_mask=None,
    need_weights=True,
    attn_mask=None,
    use_separate_proj_weight=False,
    q_proj_weight=None,
    k_proj_weight=None,
    v_proj_weight=None,
    static_k=None,
    static_v=None):
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
          will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        - attn_output_weights_all_heads: :math:`(N, H, L, S)` where N is the batch size, H is the `num_heads`,
          L is the target sequence length, S is the source sequence length.
    """
    tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v, out_proj_weight, out_proj_bias)
    if has_torch_function(tens_ops):
        return handle_torch_function(
            multi_head_attention_forward,
            tens_ops,
            query,
            key,
            value,
            embed_dim_to_check,
            num_heads,
            in_proj_weight,
            in_proj_bias,
            bias_k,
            bias_v,
            add_zero_attn,
            dropout_p,
            out_proj_weight,
            out_proj_bias,
            training=training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=use_separate_proj_weight,
            q_proj_weight=q_proj_weight,
            k_proj_weight=k_proj_weight,
            v_proj_weight=v_proj_weight,
            static_k=static_k,
            static_v=static_v,
        )
    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    # allow MHA to have different sizes for the feature dimension
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    if not use_separate_proj_weight:
        if (query is key or torch.equal(query, key)) and (key is value or torch.equal(key, value)):
            # self-attention
            q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

        elif key is value or torch.equal(key, value):
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = linear(key, _w, _b).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = linear(value, _w, _b)
    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)

        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)

        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)

        if in_proj_bias is not None:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim : (embed_dim * 2)])
            v = linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2) :])
        else:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = linear(value, v_proj_weight_non_opt, in_proj_bias)
    q = q * scaling

    if attn_mask is not None:
        assert (
            attn_mask.dtype == torch.float32
            or attn_mask.dtype == torch.float64
            or attn_mask.dtype == torch.float16
            or attn_mask.dtype == torch.uint8
            or attn_mask.dtype == torch.bool
        ), "Only float, byte, and bool types are supported for attn_mask, not {}".format(attn_mask.dtype)
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)

        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError("The size of the 2D attn_mask is not correct.")
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError("The size of the 3D attn_mask is not correct.")
        else:
            raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
        # attn_mask's dim is 3 now.

    # convert ByteTensor key_padding_mask to bool
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn(
            "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
        )
        key_padding_mask = key_padding_mask.to(torch.bool)

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_output_weights.masked_fill_(attn_mask, float("-inf"))
        else:
            attn_output_weights += attn_mask

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float("-inf"),
        )
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

    attn_output_weights = softmax(attn_output_weights, dim=-1)
    attn_output_weights = dropout(attn_output_weights, p=dropout_p, training=training)

    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        # AND (monkey patching) also return the non-averaged weights too
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, torch.sum(attn_output_weights, dim=1) / num_heads, attn_output_weights
    else:
        return attn_output, None, None

setattr(F, 'multi_head_attention_forward', alps_multi_head_attention_forward)

class BagOfWords(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size=512,
        reduce="sum",
        num_layers=2,
        activation="ReLU",
        dropout=0
    ):
        super(BagOfWords, self).__init__()

        self.emb_dim = hidden_size

        self.reduce = reduce
        assert(self.reduce in ["sum", "mean", "max"])

        self.hidden_size = hidden_size
        self.activation = getattr(nn, activation)()
        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=PAD_IDX)
        self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList([nn.Linear(self.emb_dim, self.hidden_size)])
        for i in range(num_layers - 1):
            self.layers.append(nn.Linear(self.hidden_size, self.hidden_size))

    def forward(self, x, xs_len=None, lang=None):
        x = self.embedding(x)

        if self.reduce == "sum":
            x = x.sum(dim=1)
        elif self.reduce == "mean":
            x = x.mean(dim=1)
        elif self.reduce == "max":
            x = x.max(dim=1)[0]

        outputs = []
        for l in self.layers:
            x = l(x)
            x = self.activation(x)
            x = self.dropout(x)
            outputs.append(x.unsqueeze(0))

        encoder_results = {
            'encoder_output': outputs[-1],
            'encoder_hidden': torch.cat(outputs, dim=0)
        }
        return encoder_results


class RNN_Encoder(nn.Module):
    """Encodes the input context."""

    def __init__(self, input_size, hidden_size, num_layers, dropout=0):
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

    def forward(self, input, hidden=None, xs_len=None, lang=None):
        """Return encoded state.
        :param input: (batch_size x seqlen) tensor of token indices.
        :param hidden: optional past hidden state
        """
        x = self.embedding(input)
        output, hidden = self.gru(x, hidden)

        encoder_results = {
            'encoder_output': output,
            'encoder_hidden': hidden
        }
        return encoder_results


class RNN_Decoder(nn.Module):
    """Generates a sequence of tokens in response to context."""

    def __init__(self, output_size, hidden_size, num_layers, dropout=0):
        """Initialize decoder.
        :param input_size: size of embedding
        :param hidden_size: size of GRU hidden layers
        :param num_layers: number of GRU layers
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size, padding_idx=PAD_IDX)
        self.dropout = nn.Dropout(p=dropout)
        self.gru = nn.GRU(
            hidden_size, hidden_size, num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(
            self,
            input,
            hidden,
            encoder_output=None,
            xs_len=None,
            context_vec=None,
            lang=None):
        """Return encoded state.
        :param input: batch_size x 1 tensor of token indices.
        :param hidden: past (e.g. encoder) hidden state
        """
        x = self.embedding(input)   # BxTxH
        # hidden: 1xBxH
        # encoder_output: BxSxH
        # encoder_output = encoder_output.transpose(0, 1)
        hidden = self.dropout(hidden)
        output, hidden = self.gru(x, hidden)
        # output: BxTxH
        # output = self.dropout(output)
        scores = self.softmax(self.out(output))

        decoder_results = {
            'decoder_output': scores,
            'decoder_hidden': hidden,
            'attention_weights': None,
            'context_vector': None
        }
        return decoder_results


class AttentionModule(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(AttentionModule, self).__init__()
        self.l1 = nn.Linear(hidden_size, output_size, bias=False)
        self.l2 = nn.Linear(hidden_size + output_size, output_size, bias=False)

    def forward(self, hidden, encoder_outs, src_lens):
        ''' hidden: bsz x hidden_size
        encoder_outs: bsz x sq_len x output_size
        src_lens: bsz

        x: bsz x output_size
        attn_score: bsz x sq_len'''

        x = self.l1(hidden)
        att_score = torch.bmm(encoder_outs,
                              x.unsqueeze(-1))  # this is bsz x seq x 1
        att_score = att_score.squeeze(-1)  # this is bsz x seq
        att_score = att_score.transpose(0, 1)

        seq_mask = self.sequence_mask(src_lens,
                                      max_len=max(src_lens).item(),
                                      device=hidden.device).transpose(0, 1)

        masked_att = seq_mask * att_score
        masked_att[masked_att == 0] = -1e10
        attn_scores = F.softmax(masked_att, dim=0)
        x = (
            attn_scores.unsqueeze(2) *
            encoder_outs.transpose(
                0,
                1)).sum(
            dim=0)
        x = torch.tanh(self.l2(torch.cat((x, hidden), dim=1)))
        return x, attn_scores

    def sequence_mask(
            self,
            sequence_length,
            max_len,
            device):
        if max_len is None:
            max_len = sequence_length.max().item()
        batch_size = sequence_length.size(0)
        seq_range = torch.arange(0, max_len).long()
        seq_range_expand = seq_range.unsqueeze(0).repeat([batch_size, 1])
        seq_range_expand = seq_range_expand.to(device)
        seq_length_expand = (sequence_length.unsqueeze(1)
                             .expand_as(seq_range_expand))
        return (seq_range_expand < seq_length_expand).float()


class AttentionDecoder(nn.Module):
    def __init__(
            self,
            output_size,
            hidden_size,
            dropout=0
        ):
        super(AttentionDecoder, self).__init__()

        self.output_size = output_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size, padding_idx=PAD_IDX)
        self.gru = nn.GRUCell(hidden_size*2, hidden_size, bias=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

        self.attention = AttentionModule(
            hidden_size, hidden_size)

    def forward(
            self,
            input,
            memory,
            encoder_output=None,
            xs_len=None,
            context_vec=None,
            lang=None):
        memory = memory.transpose(0, 1)
        emb = self.embedding(input)
        # emb = F.relu(emb)

        emb = emb.transpose(0, 1)
        return_scores = torch.empty(
            emb.size(0),
            emb.size(1),
            self.output_size).to(
            input.device)

        if context_vec is None:
            context_vec = torch.zeros(
                [emb.size(1), self.hidden_size]).to(emb.device)

        attn_wts_list = []

        for t in range(emb.size(0)):
            current_vec = emb[t]

            current_vec = torch.cat([current_vec, context_vec], dim=1)
            selected_memory = memory[:, 0, :]

            mem_out = self.gru(current_vec, selected_memory)

            context_vec, weights = self.attention(
                mem_out, encoder_output, xs_len)

            scores = self.out(context_vec)
            attn_wts_list.append(weights.transpose(0, 1))

            scores = self.softmax(scores)
            return_scores[t] = scores

            memory = mem_out[:, None, :]

        decoder_results = {
            'decoder_output': return_scores.transpose(0, 1).contiguous(),
            'decoder_hidden': memory.transpose(0, 1),
            'attention_weights': torch.stack(attn_wts_list, axis=1),
            'context_vector': context_vec
        }
        return decoder_results


class TransformerEncoder(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            num_layers=1,
            dropout=0,
            heads=4,
            normalize_before=False
        ):
        super(TransformerEncoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=PAD_IDX)
        self.embed_positions = PositionalEncoding(hidden_size)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_size, heads, hidden_size, dropout, normalize_before=normalize_before)
            for _ in range(num_layers)
        ])
    
    def forward(self, input, hidden=None, xs_len=None, lang=None):
        x = self.embedding(input)
        mask = torch.arange(input.size(1))[None, :].to(xs_len.device) >= xs_len[:, None]
        # shape: (batch_size, src_len)

        x = x.transpose(0, 1)   # src_len first
        x += self.embed_positions(x)
        
        outputs = []
        self_attn_weights_list = []
        for i, layer in enumerate(self.layers):
            x, self_attn_weights = layer(
                x,
                src_key_padding_mask=mask
            )
            outputs.append(x.transpose(0, 1))
            self_attn_weights_list.append(self_attn_weights)

        encoder_results = {
            'encoder_output': outputs[-1],
            'encoder_hidden': torch.cat(outputs, dim=0),
            'self_attention_weights_for_all_layers': self_attn_weights_list
        }
        return encoder_results


class TransformerDecoder(nn.Module):
    def __init__(
            self,
            output_size,
            hidden_size,
            num_layers=1,
            dropout=0,
            heads=4,
            normalize_before=False
        ):
        super(TransformerDecoder, self).__init__()

        self.output_size = output_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size, padding_idx=PAD_IDX)
        self.embed_positions = PositionalEncoding(hidden_size)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(hidden_size, heads, hidden_size, dropout, normalize_before=normalize_before)
            for _ in range(num_layers)
        ])
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(
            self,
            input,
            hidden=None,
            encoder_output=None,
            xs_len=None,
            context_vec=None,
            lang=None):
        """
        input: Tensor of shape (batch_size, src_len)
        encoder_output: Tensor of shape (batch_size, src_len, hidden_size)
        xs_len: Tensor of shape (batch_size,)
        """

        x = self.embedding(input)
        
        size = encoder_output.size(1)
        memory_mask = torch.arange(size)[None, :].to(xs_len.device) >= xs_len[:, None]
        # shape: (batch_size, src_len)

        if not self.training:
            tgt_mask = None
        else:
            size = input.size(1)
            tgt_mask = (torch.triu(torch.ones(size, size)) != 1).transpose(0, 1)
            tgt_mask = tgt_mask.to(xs_len.device)   # shape: (tgt_len, tgt_len)

        x = x.transpose(0, 1)   # src_len first
        encoder_output = encoder_output.transpose(0, 1)   # src_len first
        
        if not self.training and context_vec is not None:
            start = context_vec[0].size(0)
        else:
            start = 0
        x += self.embed_positions(x, start)
        
        if context_vec is None:
            context_vec = [None] * len(self.layers)

        self_attn_weights_list = []

        for i, layer in enumerate(self.layers):
            if context_vec[i] is None:
                context_vec[i] = x
            else:
                context_vec[i] = torch.cat([context_vec[i], x], dim=0)

            x, self_attn_weights, attn_weights = layer(
                x,
                encoder_output,
                memory_key_padding_mask=memory_mask,
                tgt_mask=tgt_mask,
                prev_states=context_vec[i]
            )
            if self.training:
                attn_weights = self_attn_weights = None

            self_attn_weights_list.append(self_attn_weights)

        scores = self.softmax(self.out(x.transpose(0, 1)))

        decoder_results = {
            'decoder_output': scores,
            'decoder_hidden': None,
            'attention_weights': attn_weights,
            'context_vector': context_vec,
            'self_attention_weights_for_all_layers': self_attn_weights_list
        }
        return decoder_results


class EncoderDecoder(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        lr=1e-3,
        use_cuda=True,
        target_dict=None,
        max_len=50,
        clip=0.3
    ):
        super(EncoderDecoder, self).__init__()

        device = torch.device(
            "cuda" if (torch.cuda.is_available() and use_cuda) else "cpu"
        )
        self.device = device
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)

        self.target_dict = target_dict

        # set up the criterion
        self.criterion = nn.NLLLoss(reduction='sum', ignore_index=PAD_IDX)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="max", min_lr=1e-6, patience=0, verbose=True)
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.5, verbose=True)
    
        self.max_len = max_len
        self.clip = clip
        self.START = torch.LongTensor([SOS_IDX]).to(device)
        self.END_IDX = EOS_IDX

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

    def scheduler_step(self, val_bleu=None):
        self.scheduler.step(val_bleu)
        # self.scheduler.step()

    def vec2txt(self, vector):
        """Convert vector to text.
        :param vector: tensor of token indices.
            1-d tensors will return a string, 2-d will return a list of strings
        """
        if vector.dim() == 1:
            output_tokens = []
            # Remove the final END_TOKEN that is appended to predictions
            for token in vector:
                if token == self.END_IDX:
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
        predicted_list = []
        real_list = []

        for data in batch_iterator:
            predicted_list += self.eval_step(data)[0]
            real_list += self.vec2txt(data['target'])

        if detokenize is not None:
            predicted_list = [detokenize(line) for line in predicted_list]
            real_list = [detokenize(line) for line in real_list]

        bleu = sacrebleu.corpus_bleu(predicted_list, [real_list], tokenize='none')
        translation_output = namedtuple('translation_output', ['score', 'output'])
        return translation_output(round(bleu.score, 2), predicted_list)

    def train_step(self, batch):
        xs, xs_len, ys, ys_len = batch['source'], batch['source_len'], batch['target'], batch['target_len']
        source_lang, target_lang = batch['source_lang'], batch['target_lang']

        if xs is None:
            return

        xs = xs.to(self.device)
        ys = ys.to(self.device)
        xs_len = xs_len.to(self.device)
        ys_len = ys_len.to(self.device)

        bsz = xs.size(0)
        starts = self.START.expand(bsz, 1)
        loss = 0
        self.zero_grad()
        self.encoder.train()
        self.decoder.train()

        encoder_results = self.encoder(xs, xs_len=xs_len, lang=source_lang)
        encoder_output = encoder_results['encoder_output']
        encoder_hidden = encoder_results['encoder_hidden']

        # Teacher forcing: Feed the target as the next input
        y_in = ys.narrow(1, 0, ys.size(1) - 1)
        decoder_input = torch.cat([starts, y_in], 1)

        decoder_results = self.decoder(
            decoder_input,
            encoder_hidden,
            encoder_output,
            xs_len,
            lang=target_lang
        )
        decoder_output = decoder_results['decoder_output']

        scores = decoder_output.view(-1, decoder_output.size(-1))
        loss = self.criterion(scores, ys.view(-1)) / ys_len.sum()
        loss.backward()
        self.update_params()

        return loss.item()

    def eval_step(self, batch):
        xs, xs_len = batch['source'], batch['source_len']
        source_lang, target_lang = batch['source_lang'], batch['target_lang']

        if xs is None:
            return

        xs = xs.to(self.device)
        xs_len = xs_len.to(self.device)

        bsz = xs.size(0)
        starts = self.START.expand(bsz, 1)  # expand to batch size
        # just predict
        self.encoder.eval()
        self.decoder.eval()
        encoder_results = self.encoder(xs, xs_len=xs_len, lang=source_lang)
        encoder_output = encoder_results['encoder_output']
        encoder_hidden = encoder_results['encoder_hidden']
        encoder_self_attn = encoder_results.get('self_attention_weights_for_all_layers')

        predictions = []
        done = [False for _ in range(bsz)]
        total_done = 0
        decoder_input = starts
        decoder_hidden = encoder_hidden

        attn_wts_list = []
        context_vec = None

        for i in range(self.max_len):
            # generate at most max_len tokens

            decoder_results = self.decoder(
                decoder_input, decoder_hidden, encoder_output, xs_len, context_vec,
                lang=target_lang
            )
            decoder_output = decoder_results['decoder_output']
            decoder_hidden = decoder_results['decoder_hidden']
            attn_wts = decoder_results['attention_weights']
            context_vec = decoder_results['context_vector']

            _max_score, preds = decoder_output.max(2)
            predictions.append(preds)
            decoder_input = preds  # set input to next step

            attn_wts_list.append(attn_wts)

            # check if we've produced the end token
            for b in range(bsz):
                if not done[b]:
                    # only add more tokens for examples that aren't done
                    if preds[b].item() == self.END_IDX:
                        # if we produced END, we're done
                        done[b] = True
                        total_done += 1
            if total_done == bsz:
                # no need to generate any more
                break
        predictions = torch.cat(predictions, 1)
        
        attn = None if attn_wts_list[0] is None else torch.cat(attn_wts_list, 1)
        return self.vec2txt(predictions), attn, encoder_self_attn


class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, *args, normalize_before=False, **kwargs):
        super(TransformerEncoderLayer, self).__init__(*args, **kwargs)
        self.normalize_before = normalize_before

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        x = src
        residual = x
        if self.normalize_before:
            x = self.norm1(x)
        x, _, self_attn_weights_all_heads = self.self_attn(
            x, x, x, attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask)
        x = residual + self.dropout1(x)
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = residual + self.dropout2(x)
        if not self.normalize_before:
            x = self.norm2(x)
        return x, self_attn_weights_all_heads


class TransformerDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, *args, normalize_before=False, **kwargs):
        super(TransformerDecoderLayer, self).__init__(*args, **kwargs)
        self.normalize_before = normalize_before

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None, prev_states=None):
        x = tgt
        residual = x
        if self.normalize_before:
            x = self.norm1(x)
        if prev_states is None:
            prev_states = x
        x, _, self_attn_weights_all_heads = self.self_attn(
            x, prev_states, prev_states, attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask)
        x = residual + self.dropout1(x)
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x, attn_weights, attn_weights_all_heads = self.multihead_attn(
            x, memory, memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)
        x = residual + self.dropout2(x)
        if not self.normalize_before:
            x = self.norm2(x)

        residual = x
        if self.normalize_before:
            x = self.norm3(x)
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = residual + self.dropout3(x)
        if not self.normalize_before:
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


class MultilingualModule(nn.Module):
    def __init__(self, modules):
        super(MultilingualModule, self).__init__()
        self.module_dict = nn.ModuleDict(modules)
    
    def forward(self, *args, lang=None, **kwargs):
        assert lang in self.module_dict, "error '{}' no module by this name".format(lang)
        return self.module_dict[lang](*args, **kwargs)
