from .token_controller import TokenMAC
import torch as th


# This multi-agent controller shares parameters between agents
class TokenLatentMAC(TokenMAC):
    def __init__(self, scheme, groups, args):
        super(TokenLatentMAC, self).__init__(scheme, groups, args)

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, self.args.n_tokens + 1, -1)

    def _build_inputs(self, batch, t):
        # currently we only support battles with marines (e.g. 3m 8m 5m_vs_6m)
        # you can implement your own with any other agent type.
        inputs = []
        raw_obs = batch["obs"][:, t]
        reshaped_obs = raw_obs.reshape(-1, self.args.n_tokens, self.args.obs_token_dim)

        inputs.append(reshaped_obs)
        inputs.append(th.zeros((reshaped_obs.size(0), 1, self.args.obs_token_dim), device=batch.device))
        inputs = th.cat(inputs, dim=1)

        b, t, e = inputs.size()
        mask = th.zeros((b, t), dtype=th.bool, device=batch.device)
        mask[:, 1:-1][inputs[:, 1:-1, self.args.obs_mask_bit] == 0] = 1
        mask = mask.repeat(1, t).reshape(b, t, t)

        mask_diag = th.ones((t, t), dtype=th.bool, device=batch.device).fill_diagonal_(0).repeat(b, 1, 1).reshape(b, t, t)
        mask = mask & mask_diag

        return inputs, mask
