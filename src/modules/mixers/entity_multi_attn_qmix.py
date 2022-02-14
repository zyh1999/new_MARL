import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.layers import MultiHeadAttention


class EntityMultiAttnQMixer(nn.Module):
    def __init__(self, args):
        super(EntityMultiAttnQMixer, self).__init__()

        self.args = args

        input_shape = args.entity_shape
        if self.args.entity_last_action:
            input_shape += args.n_actions

        self.entity_embedding_w_1 = nn.Linear(input_shape, args.mix_emb_dim)
        self.attn_w_1 = MultiHeadAttention(args.mix_emb_dim, args.n_heads)
        self.hyper_w_1 = nn.Linear(args.mix_emb_dim, args.mix_emb_dim)

        self.entity_embedding_b_1 = nn.Linear(input_shape, args.mix_emb_dim)
        self.attn_b_1 = MultiHeadAttention(args.mix_emb_dim, args.n_heads)
        self.hyper_b_1 = nn.Linear(args.mix_emb_dim, args.mix_emb_dim)

        self.entity_embedding_w_2 = nn.Linear(input_shape, args.mix_emb_dim)
        self.attn_w_2 = MultiHeadAttention(args.mix_emb_dim, args.n_heads)
        self.hyper_w_2 = nn.Linear(args.mix_emb_dim, args.mix_emb_dim)

        self.entity_embedding_b_2 = nn.Linear(input_shape, args.mix_emb_dim)
        self.attn_b_2 = MultiHeadAttention(args.mix_emb_dim, args.n_heads)
        self.hyper_b_2 = nn.Linear(args.mix_emb_dim, args.mix_emb_dim)

    def forward(self, agent_qs, states):
        # agent_qs: [batch_size, seq_length, n_agents]
        # entities: [batch_size, seq_length, n_tokens, state_token_dim]
        # entity_mask: [batch_size, seq_length, n_tokens]

        entities, entity_mask = states
        b, s, t, e = entities.shape
        entities = entities.reshape(b * s, t, e)
        entity_mask = entity_mask.reshape(b * s, t)
        agent_mask = entity_mask[:, :self.args.n_agents]
        agent_qs = agent_qs.view(b * s, 1, self.args.n_agents)

        x = F.relu(self.entity_embedding_w_1(entities))
        w_1 = self.attn_w_1(x, entity_mask.repeat(1, t).reshape(b * s, t, t))[:, :self.args.n_agents]
        w_1 = w_1.masked_fill(agent_mask.unsqueeze(2), 0)    # [b * s, n, e]
        w_1 = torch.softmax(self.hyper_w_1(w_1), dim=-1)    # [b * s, n, e]

        x = F.relu(self.entity_embedding_b_1(entities))
        b_1 = self.attn_b_1(x, entity_mask.repeat(1, t).reshape(b * s, t, t))[:, :self.args.n_agents]
        b_1 = b_1.masked_fill(agent_mask.unsqueeze(2), 0)    # [b * s, n, e]
        b_1 = self.hyper_b_1(b_1).masked_fill(agent_mask.unsqueeze(2), 0).mean(1, True)  # [b * s, 1, e]

        h = F.elu(torch.bmm(agent_qs, w_1) + b_1)  # [b * s, 1, n] x [b * s, n, e] + [b * s, 1, e] = [b * s, 1, e]

        x = F.relu(self.entity_embedding_w_2(entities))
        w_2 = self.attn_w_2(x, entity_mask.repeat(1, t).reshape(b * s, t, t))[:, :self.args.n_agents]
        w_2 = w_2.masked_fill(agent_mask.unsqueeze(2), 0)    # [b * s, n, e]
        w_2 = torch.softmax(self.hyper_w_2(w_2), dim=-1).masked_fill(agent_mask.unsqueeze(2), 0).mean(1, True)  # [b * s, 1, e]

        x = F.relu(self.entity_embedding_b_2(entities))
        b_2 = self.attn_b_2(x, entity_mask.repeat(1, t).reshape(b * s, t, t))[:, :self.args.n_agents]
        b_2 = b_2.masked_fill(agent_mask.unsqueeze(2), 0)    # [b * s, n, e]
        b_2 = self.hyper_b_2(b_2).masked_fill(agent_mask.unsqueeze(2), 0).mean((1, 2), True)  # [b * s, 1, 1]

        q_tot = torch.bmm(h, w_2.transpose(1, 2)) + b_2  # [b * s, 1, e] x [b * s, e, 1] + [b * s, 1, 1] = [b * s, 1, 1]

        q_tot = q_tot.view(b, s, 1)  # [batch_size, max_seq_length - 1, 1]

        return q_tot
