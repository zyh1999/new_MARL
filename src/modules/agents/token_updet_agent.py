import torch.nn as nn
import torch.nn.functional as F
import torch
import argparse


class UPDeT(nn.Module):
    def __init__(self, input_shape, args):
        super(UPDeT, self).__init__()
        self.args = args
        self.transformer = Transformer(input_shape, args.emb_dim, args.heads, args.depth, args.emb_dim)
        self.q_basic = nn.Linear(args.emb_dim, args.n_fixed_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return torch.zeros(1, self.args.emb_dim).cuda()

    def forward(self, inputs, hidden_state):
        # x: [batch_size * n_agents, n_tokens, obs_token_dim]
        # mask: [batch_size * n_agents, n_tokens, n_tokens]
        # hidden_state: [batch_size, n_agents, 1, emb_dim]

        x, mask = inputs
        h_in = hidden_state.reshape(-1, 1, self.args.emb_dim)
        outputs, _ = self.transformer.forward(x, h_in, mask)
        # first output for 6 action (no_op stop up down left right)
        q_basic_actions = self.q_basic(outputs[:, 0, :])

        # last dim for hidden state
        h = outputs[:, -1:, :]

        q = q_basic_actions

        if self.args.action_enemy_wise:
            q_enemies = self.q_basic(outputs[:, 1:self.args.n_enemies + 1, :]).mean(2)
            q = torch.cat((q, q_enemies), 1)

        if self.args.action_ally_wise:
            q_allies = self.q_basic(outputs[:, -(self.args.n_agents - 1):, :]).mean(2)
            q = torch.cat((q, q_allies), 1)

        return q, h


class SelfAttention(nn.Module):
    def __init__(self, emb, heads=8, mask=False):

        super().__init__()

        self.emb = emb
        self.heads = heads
        self.mask = mask

        self.tokeys = nn.Linear(emb, emb * heads, bias=False)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False)
        self.tovalues = nn.Linear(emb, emb * heads, bias=False)

        self.unifyheads = nn.Linear(heads * emb, emb)

    def forward(self, x, mask):
        # x: [batch_size * n_agents, n_units + 1, emb]
        b, t, e = x.size()
        h = self.heads
        keys = self.tokeys(x).view(b, t, h, e)  # keys: [batch_size * n_agents, n_units + 1, heads, emb]
        queries = self.toqueries(x).view(b, t, h, e)
        values = self.tovalues(x).view(b, t, h, e)

        # compute scaled dot-product self-attention

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)  # keys: [batch_size * n_agents * heads, n_units + 1, emb]
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        queries = queries / (e ** (1 / 4))
        keys = keys / (e ** (1 / 4))
        # - Instead of dividing the dot products by sqrt(e), we scale the keys and values.
        #   This should be more memory efficient

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))  # dot: [batch_size * n_agents * heads, n_units + 1, n_units + 1]

        assert dot.size() == (b * h, t, t)

        if self.mask:  # mask out the upper half of the dot matrix, excluding the diagonal
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        if mask is not None:
            dot = dot.masked_fill(mask == 0, -1e9)

        dot = F.softmax(dot, dim=2)  # dot: [batch_size * n_agents * heads, n_units + 1, n_units + 1]
        # - dot now has row-wise self-attention probabilities

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t, e)  # out: [batch_size * n_agents, heads, n_units + 1, emb]

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)  # out: [batch_size * n_agents, n_units + 1, heads * emb]

        return self.unifyheads(out)   # [batch_size * n_agents, n_units + 1, emb]

class TransformerBlock(nn.Module):

    def __init__(self, emb, heads, mask, ff_hidden_mult=4, dropout=0.0):
        super().__init__()

        self.attention = SelfAttention(emb, heads=heads, mask=mask)
        self.mask = mask

        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )

        self.do = nn.Dropout(dropout)

    def forward(self, x_mask):

        x, mask = x_mask  # x: [batch_size * n_agents, n_units + 1, emb]

        attended = self.attention(x, mask)  # [batch_size * n_agents, n_units + 1, emb]

        x = self.norm1(attended + x)

        x = self.do(x)

        fedforward = self.ff(x)

        x = self.norm2(fedforward + x)

        x = self.do(x)

        return x, mask    # x: [batch_size * n_agents, n_units + 1, emb]


class Transformer(nn.Module):

    def __init__(self, input_dim, emb, heads, depth, output_dim):
        super().__init__()

        self.num_tokens = output_dim

        self.token_embedding = nn.Linear(input_dim, emb)

        tblocks = []
        for i in range(depth):
            tblocks.append(
                TransformerBlock(emb=emb, heads=heads, mask=False))

        self.tblocks = nn.Sequential(*tblocks)

        self.toprobs = nn.Linear(emb, output_dim)

    def forward(self, x, h, mask):
        # x: [batch_size * n_agents, n_units, n_unit_features]
        # h: [batch_size * n_agents, 1, emb]
        # mask: None

        tokens = self.token_embedding(x)  # tokens: [batch_size * n_agents, n_units, emb]
        tokens = torch.cat((tokens, h), 1)  # tokens: [batch_size * n_agents, n_units + 1, emb]

        b, t, e = tokens.size()  # batch_size * n_agents, n_units + 1, emb

        x, mask = self.tblocks((tokens, mask))  # x: [batch_size * n_agents, n_units + 1, emb]

        x = self.toprobs(x.view(b * t, e)).view(b, t, self.num_tokens)  # x: [batch_size * n_agents, n_units + 1, output_dim]

        return x, tokens

def mask_(matrices, maskval=0.0, mask_diagonal=True):

    b, h, w = matrices.size()
    indices = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
    matrices[:, indices[0], indices[1]] = maskval
