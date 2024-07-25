import math
from typing import List, Optional, Tuple
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.init import constant_
from torch.nn.init import xavier_uniform_
import warnings

Tensor = torch.Tensor


class FPL(nn.Module):
    def __init__(self, num_embeddings, num_channel, embed_dim, filter_sizes, pad_idx=0):
        super(FPL, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embed_dim, padding_idx=pad_idx
        )
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1,
                                              out_channels=num_channel,
                                              kernel_size=(filter_size, embed_dim))
                                    for filter_size in filter_sizes])
        self.re_convs = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, len(filter_sizes)),
                                  stride=(1, len(filter_sizes)))

    def forward(self, x):
        # [ batch_size, seq_len ]
        #print(x.shape)
        x = self.embedding(x).unsqueeze(1)
        #print(x.shape)
        # [ batch_size, 1, seq_len, embed_dim ]
        
        conved = [F.relu(conv(x).squeeze(3)) for conv in self.convs]
        #print(conved[0].shape)
        # conved[i] = [ batch_size, num_filters, conved_height ]
        # conved_width equals to 1, and is squeezed

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        #print(pooled[0].shape)  
        # pooled[i] = [ batch_size, num_filters ]
        # pooled_height equals to 1, and is squeezed

        # [batch_size,len(filter_size),num_filters]

        x = torch.cat(pooled, 0)
        #print(x.shape)
        x = x.unsqueeze(0)
        #print(x.shape)
        return x


def _in_projection_packed(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        w: Tensor,
        b: Optional[Tensor] = None,
) -> List[Tensor]:
    
    E = q.size(-1)
 
    if k is v:
        if q is k:
            # self
            return nn.functional.linear(q, w, b).chunk(3, dim=-1)
        else:
            # seq2seq
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            return (nn.functional.linear(q, w_q, b_q),) + nn.functional.linear(k, w_kv, b_kv).chunk(2, dim=-1)
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return nn.functional.linear(q, w_q, b_q), nn.functional.linear(k, w_k, b_k), nn.functional.linear(v, w_v, b_v)


def _scaled_dot_product_attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Optional[Tensor] = None,
        dropout_p: float = 0.0,
) -> Tuple[Tensor, Tensor]:

    B, Nt, E = q.shape
    q = q / math.sqrt(E)
    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    attn = torch.bmm(q, k.transpose(-2, -1))
    if attn_mask is not None:
        attn += attn_mask
    # attn
    attn = nn.functional.softmax(attn, dim=-1)
    if dropout_p > 0.0:
        attn = nn.functional.dropout(attn, p=dropout_p)
    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    output = torch.bmm(attn, v)
    return output, attn


def multi_head_attention_forward(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        num_heads: int,
        in_proj_weight: Tensor,
        in_proj_bias: Optional[Tensor],
        dropout_p: float,
        out_proj_weight: Tensor,
        out_proj_bias: Optional[Tensor],
        training: bool = True,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        use_separate_proj_weight: bool = False,
        q_proj_weight: Optional[Tensor] = None,
        k_proj_weight: Optional[Tensor] = None,
        v_proj_weight: Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor]]:


    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape
    head_dim = embed_dim // num_heads
    q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)

    if attn_mask is not None:
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)
        else:
            assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, \
                f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"

        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(
                    f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(
                    f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
        else:
            raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn(
            "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
        key_padding_mask = key_padding_mask.to(torch.bool)


    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (bsz, src_len), \
            f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len). \
            expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
        if attn_mask is None:
            attn_mask = key_padding_mask
        elif attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.logical_or(key_padding_mask)
        else:
            attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

    if attn_mask is not None and attn_mask.dtype == torch.bool:
        new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
        new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        attn_mask = new_attn_mask


    if not training:
        dropout_p = 0.0
    attn_output, attn_output_weights = _scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = nn.functional.linear(attn_output, out_proj_weight, out_proj_bias)
    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None


class MultiheadAttention(nn.Module):


    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, kdim=None, vdim=None,
                 batch_first=False) -> None:
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.empty((embed_dim, embed_dim)))
            self.k_proj_weight = Parameter(torch.empty((embed_dim, self.kdim)))
            self.v_proj_weight = Parameter(torch.empty((embed_dim, self.vdim)))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim)))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight)
        else:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)
        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights


class TransformerEncoderLayer(nn.Module):


    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=False) -> None:
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.activation = activation
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):

    # the combined model of textual features and statistical features
    def __init__(self, FPL, encoder_layer, num_layers, filter_sizes, num_channel, output_dim, embed_dim, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layer = encoder_layer
        self.num_layers = num_layers
        self.norm = norm
        self.input_dim = len(filter_sizes) * num_channel
        self.feature1 = nn.Linear(32, 128)
        self.feature2 = nn.Linear(128, 32)
        self.fc = nn.Linear(self.input_dim+32, 128)
        self.output = nn.Linear(128, output_dim)
        
        # [2, 3, 4, 5, 6]
        self.positional_encoding = FPL(num_embeddings=50285, num_channel=num_channel, embed_dim=embed_dim,
                                       filter_sizes=filter_sizes)


    
    
    def forward(self, src: Tensor, features, mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = self.positional_encoding(src)

        residual = output

        for _ in range(self.num_layers):
            out = self.layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            out = self.norm(out)

        out = out + residual
        out = out.reshape(-1, self.input_dim)
        
        features = features.to(torch.float32)
        #print(features.dtype)
        features = self.feature1(features)
        features = F.relu(features)
        features = self.feature2(features)
        out = torch.cat((out, features),dim=1)
        out = F.normalize(out, dim=-1)
        #print(out.shape)
        
        out = self.fc(out)
        out = F.relu(out)
        out = self.output(out)
        out = F.softmax(out, dim=-1)
        return out

    

class GPAResNet(nn.Module):
    def __init__(self, d_model, ndead, batch_first, FPL=FPL):
        super(GPAResNet, self).__init__()
        self.encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=ndead, batch_first=batch_first)
        self.transformer_encoder = TransformerEncoder(FPL=FPL, encoder_layer=self.encoder_layer, num_layers=6,
                                                      filter_sizes=[2, 3, 5, 7, 11], num_channel=200, output_dim=2,
                                                      embed_dim=200)

    def forward(self, x, f):
        x = self.transformer_encoder(x,f)
        return x

if __name__ == '__main__':
    # test1
    # encoder_layer = TransformerEncoderLayer(d_model=200, nhead=20,batch_first=True)
    # transformer_encoder = TransformerEncoder(encoder_layer, num_layers=6, input_dim=1000 ,output_dim=2)
    # src = torch.randn((10, 32, 200))
    # out = transformer_encoder(src)
    # print(out.shape)

    # test2
    from transformers import AutoTokenizer

    transformer_encoder = GPAResNet(d_model=200, ndead=20, batch_first=True, num_layers=6, input_dim=1000, out_dim=2)
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
    test_file_path = r'.\your_path'
    with open(test_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        txt = f.read()
    token = tokenizer(txt, padding=True, return_tensors='pt')
    ids = token['input_ids']
    out_put = transformer_encoder(ids)
    print(out_put.shape)
