import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


class MLP(nn.Module):
    def __init__(self, emb_type, model_in_size, d_model, e_layers):
        super(MLP, self).__init__()
        self.emb_type = emb_type
        self.model_in_size = model_in_size
        self.d_model = d_model
        self.e_layers = e_layers

        self.mlp = nn.Sequential(
            nn.Linear(self.model_in_size, self.d_model), GELUComplex() if self.emb_type=='freq' else nn.GELU(),
            *[nn.Sequential(nn.Linear(self.d_model, self.d_model), GELUComplex() if self.emb_type=='freq' else nn.GELU()) \
              for _ in range(self.e_layers+2)],
            nn.Linear(self.d_model, self.model_in_size))
        
    def forward(self, x):
        return self.mlp(x)
    

class RNN(nn.Module):
    def __init__(self, emb_type, model_in_size, d_model, e_layers, dropout):
        super(RNN, self).__init__()
        self.emb_type = emb_type
        self.model_in_size = model_in_size
        self.d_model = d_model
        self.e_layers = e_layers
        self.dropout = dropout

        self.rnn = nn.GRU(input_size=self.model_in_size, hidden_size=self.d_model, num_layers=self.e_layers, \
                          batch_first=True, dropout=self.dropout, bidirectional=True)
        self.linear = nn.Linear(in_features=2*self.d_model, out_features=self.model_in_size)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.linear(x)
        return x
    

class Transformer(nn.Module):
    def __init__(self, emb_type, model_in_size, d_ff, e_layers, n_heads, activation, dropout=0.1, factor=5):
        super(Transformer, self).__init__()
        self.emb_type = emb_type
        self.model_in_size = model_in_size
        self.d_ff = d_ff
        self.e_layers = e_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.factor = factor
        self.activation = activation


        self.trans = Encoder([EncoderLayer(AttentionLayer(FullAttention(False, self.factor, attention_dropout=self.dropout, output_attention=False), self.model_in_size, self.n_heads),
            self.model_in_size, self.d_ff, dropout=self.dropout, activation=self.activation) for l in range(self.e_layers)], 
            norm_layer=nn.Sequential(Transpose(1,2), nn.BatchNorm1d(self.model_in_size), Transpose(1,2)) if self.emb_type=='patch' else torch.nn.LayerNorm(self.model_in_size))

    def forward(self, x):
        x = self.trans(x)
        return x


# Modules for Transformer
class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale*scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x
    

class GELUComplex(nn.Module):
    def __init__(self):
        super(GELUComplex, self).__init__()
        self.gelu_real = nn.GELU()
        self.gelu_img = nn.GELU()

    def forward(self, x):
        gelu_real = self.gelu_real(x.real)
        gelu_imag = self.gelu_img(x.imag)
        return torch.complex(gelu_real, gelu_imag)