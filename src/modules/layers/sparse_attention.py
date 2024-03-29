import torch
import torch.nn as nn
#from entmax import sparsemax


class ScaledDotProductSparseAttention(nn.Module):
    def __init__(self, temperature, dropout_attn=0.0):
        super().__init__()
        self.temperature = temperature
        self.dropout_attn = nn.Dropout(dropout_attn)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q / self.temperature, k.transpose(1, 2))

        if mask is not None:
            attn = attn.masked_fill(mask.bool(), float('-inf'))
            # sparsemax cannot handle all -inf input
            inf_mask = mask.bool().all(dim=2).unsqueeze(2).repeat(1, 1, mask.size()[2])
            attn = attn.masked_fill(inf_mask, 0)

        attn = sparsemax(attn, dim=2)

        if mask is not None:
            # if agent is inactive and all entities were masked
            attn = attn.masked_fill(inf_mask, 0)

        attn = self.dropout_attn(attn)

        out = torch.bmm(attn, v)

        return out, attn


class MultiHeadSparseAttention(nn.Module):
    def __init__(self, emb_dim, n_heads, dropout_attn=0.0, dropout_attn_out=0.0):
        super().__init__()

        self.emb_dim = emb_dim
        self.n_heads = n_heads

        self.w_k = nn.Linear(emb_dim, n_heads * emb_dim, bias=False)
        self.w_q = nn.Linear(emb_dim, n_heads * emb_dim, bias=False)
        self.w_v = nn.Linear(emb_dim, n_heads * emb_dim, bias=False)

        self.attention = ScaledDotProductSparseAttention(temperature=emb_dim ** 0.5, dropout_attn=dropout_attn)
        self.dropout_attn_out = nn.Dropout(dropout_attn_out)

        self.fc = nn.Linear(n_heads * emb_dim, emb_dim)

    def forward(self, x, mask=None):
        b, t, e = x.size()
        n_heads = self.n_heads

        k = self.w_k(x).view(b, t, n_heads, e)
        q = self.w_q(x).view(b, t, n_heads, e)
        v = self.w_v(x).view(b, t, n_heads, e)

        k = k.permute(2, 0, 1, 3).contiguous().view(n_heads * b, t, e)
        q = q.permute(2, 0, 1, 3).contiguous().view(n_heads * b, t, e)
        v = v.permute(2, 0, 1, 3).contiguous().view(n_heads * b, t, e)

        if mask is not None:
            mask = mask.repeat(n_heads, 1, 1)

        out, _ = self.attention(q, k, v, mask=mask)

        out = out.view(n_heads, b, t, e)
        out = out.permute(1, 2, 0, 3).contiguous().view(b, t, n_heads * e)

        out = self.fc(out)
        out = self.dropout_attn_out(out)

        return out


class PositionWiseFeedForward(nn.Module):
    def __init__(self, emb_dim, ff_emb_dim, dropout_ff=0.0):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(emb_dim, ff_emb_dim),
            nn.ReLU(),
            nn.Linear(ff_emb_dim, emb_dim)
        )

        self.dropout = nn.Dropout(dropout_ff)

    def forward(self, x):
        out = self.fc(x)
        out = self.dropout(out)

        return out


class SparseTransformerBlock(nn.Module):

    def __init__(self, emb_dim, n_heads, ff_emb_dim, dropout_attn=0.0, dropout_attn_out=0.0, dropout_ff=0.0):
        super().__init__()

        self.attn = MultiHeadSparseAttention(emb_dim, n_heads, dropout_attn, dropout_attn_out)
        self.ff = PositionWiseFeedForward(emb_dim, ff_emb_dim, dropout_ff)

        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)

    def forward(self, inputs):
        x, mask = inputs

        attn_x = self.attn(x, mask)
        x = self.norm1(attn_x + x)

        ff_x = self.ff(x)
        x = self.norm2(ff_x + x)

        return x, mask


class SparseTransformer(nn.Module):

    def __init__(self, n_blocks, emb_dim, n_heads, ff_emb_dim,
                 dropout_attn=0.0, dropout_attn_out=0.0, dropout_ff=0.0):
        super().__init__()

        self.transformer_blocks = nn.Sequential(*[
            SparseTransformerBlock(emb_dim, n_heads, ff_emb_dim, dropout_attn, dropout_attn_out, dropout_ff)
            for _ in range(n_blocks)])

        self.fc = nn.Linear(emb_dim, emb_dim)

    def forward(self, x, mask=None):

        out, _ = self.transformer_blocks((x, mask))

        out = self.fc(out)

        return out
