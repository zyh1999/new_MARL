import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.layers import MultiHeadAttention


class TokenWiseBranchAttnAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(TokenWiseBranchAttnAgent, self).__init__()
        self.args = args

        self.token_embedding = nn.Linear(input_shape, args.emb_dim)

        self.attn = MultiHeadAttention(args.emb_dim, args.n_heads)
        self.fc_attn = nn.Linear(args.emb_dim, args.rnn_hidden_dim)

        self.fc_rnn = nn.Linear(args.emb_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)

        self.fc = nn.Linear(args.rnn_hidden_dim * 2, args.n_fixed_actions)

    def init_hidden(self):
        return self.fc_rnn.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        # x: [batch_size * n_agents, n_tokens, obs_token_dim]
        # mask: [batch_size * n_agents, n_tokens, n_tokens]
        # hidden_state: [batch_size, n_agents, n_tokens, rnn_hidden_dim]

        x, mask = inputs
        b, t, e = x.size()

        x = F.relu(self.token_embedding(x))

        x1 = self.attn(x, mask)
        x1 = F.relu(self.fc_attn(x1))

        x2 = F.relu(self.fc_rnn(x)).reshape(-1, self.args.rnn_hidden_dim)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x2, h_in).reshape(b, t, self.args.rnn_hidden_dim)

        x3 = torch.cat((x1, h), dim=-1)

        q_self_actions = self.fc(x3[:, 0, :])

        q = q_self_actions
        if self.args.action_enemy_wise:
            q_mutual_actions = self.fc(x3[:, 1:self.args.n_enemies + 1, :]).mean(2)
            q = torch.cat((q, q_mutual_actions), 1)
        if self.args.action_ally_wise:
            q_mutual_actions = self.fc(x3[:, -(self.args.n_agents - 1):, :]).mean(2)
            q = torch.cat((q, q_mutual_actions), 1)

        return q, h
