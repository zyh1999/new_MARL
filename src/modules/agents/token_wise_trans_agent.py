import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.layers import Transformer


class TokenWiseTransAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(TokenWiseTransAgent, self).__init__()
        self.args = args

        self.token_embedding = nn.Linear(input_shape, args.emb_dim)
        self.transformer = Transformer(args.n_blocks, args.emb_dim, args.n_heads, args.emb_dim * 4)

        self.fc1 = nn.Linear(args.emb_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_fixed_actions)

    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        # x: [batch_size * n_agents, n_tokens, obs_token_dim]
        # mask: [batch_size * n_agents, n_tokens, n_tokens]
        # hidden_state: [batch_size, n_agents, n_tokens, rnn_hidden_dim]

        x, mask = inputs
        b, t, e = x.size()

        x = F.relu(self.token_embedding(x))
        x = self.transformer.forward(x, mask)

        x = F.relu(self.fc1(x)).reshape(-1, self.args.rnn_hidden_dim)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in).reshape(b, t, self.args.rnn_hidden_dim)

        q_self_actions = self.fc2(h[:, 0, :])

        q = q_self_actions
        if self.args.action_enemy_wise:
            q_mutual_actions = self.fc2(h[:, 1:self.args.n_enemies + 1, :]).mean(2)
            q = torch.cat((q, q_mutual_actions), 1)
        if self.args.action_ally_wise:
            q_mutual_actions = self.fc2(h[:, -(self.args.n_agents - 1):, :]).mean(2)
            q = torch.cat((q, q_mutual_actions), 1)

        return q, h


