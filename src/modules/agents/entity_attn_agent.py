import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.layers import MultiHeadAttention


class EntityAttnAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(EntityAttnAgent, self).__init__()
        self.args = args

        self.entity_embedding = nn.Linear(input_shape, args.emb_dim)
        self.attn = MultiHeadAttention(args.emb_dim, args.n_heads)
        self.fc1 = nn.Linear(args.emb_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        # entities: [batch_size, seq_length, n_tokens, obs_token_dim]
        # obs_mask: [batch_size, seq_length, n_tokens, n_tokens]
        # entity_mask: [batch_size, seq_length, n_tokens]
        # hidden_state: [batch_size, n_agents, rnn_hidden_dim]

        entities, obs_mask, entity_mask = inputs
        b, s, t, e = entities.shape
        entities = entities.reshape(b * s, t, e)
        obs_mask = obs_mask.reshape(b * s, t, t)
        entity_mask = entity_mask.reshape(b * s, t)
        agent_mask = entity_mask[:, :self.args.n_agents]

        x = F.relu(self.entity_embedding(entities))
        x = self.attn(x, obs_mask)[:, :self.args.n_agents]
        x = x.masked_fill(agent_mask.unsqueeze(2), 0)

        x = F.relu(self.fc1(x))
        x = x.reshape(b, s, self.args.n_agents, -1)

        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = []
        for i in range(s):
            h_in = self.rnn(x[:, i].reshape(-1, self.args.rnn_hidden_dim), h_in)
            h.append(h_in.reshape(b, self.args.n_agents, self.args.rnn_hidden_dim))
        h = torch.stack(h, dim=1)

        q = self.fc2(h)
        q = q.reshape(b, s, self.args.n_agents, -1)
        q = q.masked_fill(agent_mask.reshape(b, s, self.args.n_agents, 1), 0)

        return q, h
