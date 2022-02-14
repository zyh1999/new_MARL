import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.layers import PatternTransformer


class TokenPatternTransQMixer(nn.Module):
    def __init__(self, args):
        super(TokenPatternTransQMixer, self).__init__()

        self.args = args
        self.state_shape = None

        self.token_embedding = nn.Linear(args.state_token_dim, args.mix_emb_dim)
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
        # states: [batch_size, seq_length, n_tokens, state_token_dim]

        self.state_shape = states.size()
        b, s, t, e = states.size()
        states = states.reshape(b * s, t, e)  # [b * s, t, e]
        agent_qs = agent_qs.view(b * s, 1, self.args.n_agents)  # [b * s, 1, n]

        x = F.relu(self.token_embedding(states))
        x = self.transformer.forward(x)[:, :self.args.n_agents]  # [b * s, n, e]

        if self.args.scale_q:
            w_0 = self.softmax(self.hyper_w_0(x).view(b * s, 1, self.args.n_agents))  # [b * s, 1, n]
            agent_qs = torch.mul(w_0, agent_qs)  # [b * s, 1, n]

        w_1 = torch.abs(self.hyper_w_1(x))  # [b * s, n, e]
        b_1 = self.hyper_b_1(x).mean(1, True)  # [b * s, 1, e]
        h = F.elu(torch.bmm(agent_qs, w_1) + b_1)  # [b * s, 1, n] x [b * s, n, e] + [b * s, 1, e] = [b * s, 1, e]

        w_2 = torch.abs(self.hyper_w_2(x)).mean(1, True)  # [b * s, 1, e]
        b_2 = self.hyper_b_2(x).mean(1, True)  # [b * s, 1, 1]
        q_tot = torch.bmm(h, w_2.transpose(1, 2)) + b_2  # [b * s, 1, e] x [b * s, e, 1] + [b * s, 1, 1] = [b * s, 1, 1]

        q_tot = q_tot.view(b, s, 1)  # [batch_size, seq_length, 1]

        return q_tot

    def get_disentangle_loss(self, mode="contrastive"):
        b, s, t, e = self.state_shape

        loss = 0
        for block in self.transformer.transformer_blocks:
            loss += block.attn.cal_disentangle_loss(mode=mode)
        loss = torch.mean(loss.reshape(b, s, t), dim=2)

        return loss

