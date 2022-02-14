import torch
import torch.nn as nn
import torch.nn.functional as F


class DyAN(nn.Module):
    def __init__(self, input_shape, args):
        super(DyAN, self).__init__()
        self.args = args

        self.token_embedding_self = nn.Linear(input_shape, args.emb_dim)
        self.token_embedding_enemy = nn.Linear(input_shape, args.emb_dim)
        self.token_embedding_ally = nn.Linear(input_shape, args.emb_dim)

        self.fc1 = nn.Linear(args.emb_dim * 3, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        # x: [batch_size * n_agents, n_tokens, obs_token_dim]
        # hidden_state: [batch_size, n_agents, rnn_hidden_dim]

        x, mask = inputs

        x_self = F.relu(self.token_embedding_self(x[:, 0, :]))
        x_enemy = F.relu(self.token_embedding_enemy(x[:, 1:self.args.n_enemies + 1, :])).sum(1)
        x_ally = F.relu(self.token_embedding_ally(x[:, -(self.args.n_agents - 1):, :])).sum(1)
        x = torch.cat([x_self, x_enemy, x_ally], dim=-1)

        x = F.relu(self.fc1(x))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)

        return q, h
