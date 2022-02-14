import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.layers import PatternTransformer


class EntityPatternTransQMixer(nn.Module):
    def __init__(self, args):
        super(EntityPatternTransQMixer, self).__init__()

        self.args = args

        input_shape = args.entity_shape
        if self.args.entity_last_action:
            input_shape += args.n_actions

        self.entity_shape = None

        self.entity_embedding = nn.Linear(input_shape, args.mix_emb_dim)
        self.transformer = PatternTransformer(args.mix_n_blocks, args.mix_emb_dim, args.mix_n_heads, args.mix_emb_dim * 4)

        if self.args.scale_q:
            self.hyper_w_0 = nn.Linear(args.mix_emb_dim, 1)
            self.softmax = nn.Softmax(dim=-1)

        self.hyper_w_1 = nn.Linear(args.mix_emb_dim, args.mix_emb_dim)
        self.hyper_b_1 = nn.Linear(args.mix_emb_dim, args.mix_emb_dim)

        self.hyper_w_2 = nn.Linear(args.mix_emb_dim, args.mix_emb_dim)
        self.hyper_b_2 = nn.Sequential(nn.Linear(args.mix_emb_dim, args.mix_emb_dim),
                                       nn.ReLU(),
                                       nn.Linear(args.mix_emb_dim, 1))

    def forward(self, agent_qs, states):
        # agent_qs: [batch_size, seq_length, n_agents]
        # entities: [batch_size, seq_length, n_tokens, state_token_dim]
        # entity_mask: [batch_size, seq_length, n_tokens]

        entities, entity_mask = states
        self.entity_shape = entities.shape
        b, s, t, e = entities.shape
        entities = entities.reshape(b * s, t, e)
        entity_mask = entity_mask.reshape(b * s, t)
        agent_mask = entity_mask[:, :self.args.n_agents]
        agent_qs = agent_qs.view(b * s, 1, self.args.n_agents)

        x = F.relu(self.entity_embedding(entities))
        x = self.transformer(x, entity_mask.repeat(1, t).reshape(b * s, t, t))[:, :self.args.n_agents]
        x = x.masked_fill(agent_mask.unsqueeze(2), 0)    # [b * s, n, e]

        if self.args.scale_q:
            w_0 = self.hyper_w_0(x).masked_fill(agent_mask.unsqueeze(2), float('-inf'))
            w_0 = self.softmax(w_0.view(-1, 1, self.args.n_agents))  # [b * s, 1, n]
            agent_qs = torch.mul(w_0, agent_qs)  # [b * s, 1, n]

        w_1 = torch.abs(self.hyper_w_1(x))  # [b * s, n, e]
        b_1 = self.hyper_b_1(x).masked_fill(agent_mask.unsqueeze(2), 0).mean(1, True)  # [b * s, 1, e]
        h = F.elu(torch.bmm(agent_qs, w_1) + b_1)  # [b * s, 1, n] x [b * s, n, e] + [b * s, 1, e] = [b * s, 1, e]

        w_2 = torch.abs(self.hyper_w_2(x)).masked_fill(agent_mask.unsqueeze(2), 0).mean(1, True)  # [b * s, 1, e]
        b_2 = self.hyper_b_2(x).masked_fill(agent_mask.unsqueeze(2), 0).mean(1, True)  # [b * s, 1, 1]
        q_tot = torch.bmm(h, w_2.transpose(1, 2)) + b_2  # [b * s, 1, e] x [b * s, e, 1] + [b * s, 1, 1] = [b * s, 1, 1]

        q_tot = q_tot.view(b, s, 1)  # [batch_size, max_seq_length - 1, 1]

        return q_tot

    def get_disentangle_loss(self, mode="contrastive"):
        b, s, t, e = self.entity_shape

        loss = 0
        for block in self.transformer.transformer_blocks:
            loss += block.attn.cal_disentangle_loss(mode=mode)
        loss = torch.mean(loss.reshape(b, s, t), dim=2)

        return loss
