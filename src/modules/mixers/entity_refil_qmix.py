import torch as th
import torch.nn as nn
import torch.nn.functional as F


class AttentionHyperNet(nn.Module):
    """
    mode='matrix' gets you a <n_agents x mixing_embed_dim> sized matrix
    mode='vector' gets you a <mixing_embed_dim> sized vector by averaging over agents
    mode='alt_vector' gets you a <n_agents> sized vector by averaging over embedding dim
    mode='scalar' gets you a scalar by averaging over agents and embed dim
    ...per set of entities
    """
    def __init__(self, args, extra_dims=0, mode='matrix'):
        super(AttentionHyperNet, self).__init__()
        self.args = args
        self.mode = mode
        self.extra_dims = extra_dims
        self.entity_dim = args.entity_shape
        if self.args.entity_last_action:
            self.entity_dim += args.n_actions
        if extra_dims > 0:
            self.entity_dim += extra_dims

        hypernet_embed = args.hypernet_embed
        self.fc1 = nn.Linear(self.entity_dim, hypernet_embed)
        if args.pooling_type is None:
            self.attn = EntityAttentionLayer(hypernet_embed,
                                             hypernet_embed,
                                             hypernet_embed, args)
        else:
            self.attn = EntityPoolingLayer(hypernet_embed,
                                           hypernet_embed,
                                           hypernet_embed,
                                           args.pooling_type,
                                           args)
        self.fc2 = nn.Linear(hypernet_embed, args.mixing_embed_dim)

    def forward(self, entities, entity_mask, attn_mask=None):
        x1 = F.relu(self.fc1(entities))
        agent_mask = entity_mask[:, :self.args.n_agents]
        if attn_mask is None:
            # create attn_mask from entity mask
            attn_mask = 1 - th.bmm((1 - agent_mask.to(th.float)).unsqueeze(2),
                                   (1 - entity_mask.to(th.float)).unsqueeze(1))
        x2 = self.attn(x1, pre_mask=attn_mask.to(th.bool),
                       post_mask=agent_mask)
        x3 = self.fc2(x2)
        x3 = x3.masked_fill(agent_mask.unsqueeze(2), 0)
        if self.mode == 'vector':
            return x3.mean(dim=1)
        elif self.mode == 'alt_vector':
            return x3.mean(dim=2)
        elif self.mode == 'scalar':
            return x3.mean(dim=(1, 2))
        return x3


class FlexQMixer(nn.Module):
    def __init__(self, args):
        super(FlexQMixer, self).__init__()
        self.args = args

        self.n_agents = args.n_agents

        self.embed_dim = args.mixing_embed_dim

        self.hyper_w_1 = AttentionHyperNet(args, mode='matrix')
        self.hyper_w_final = AttentionHyperNet(args, mode='vector')
        self.hyper_b_1 = AttentionHyperNet(args, mode='vector')
        # V(s) instead of a bias for the last layers
        self.V = AttentionHyperNet(args, mode='scalar')

        self.non_lin = F.elu
        if getattr(self.args, "mixer_non_lin", "elu") == "tanh":
            self.non_lin = F.tanh

    def forward(self, agent_qs, inputs, imagine_groups=None):
        entities, entity_mask = inputs
        bs, max_t, ne, ed = entities.shape

        entities = entities.reshape(bs * max_t, ne, ed)
        entity_mask = entity_mask.reshape(bs * max_t, ne)
        if imagine_groups is not None:
            agent_qs = agent_qs.view(-1, 1, self.n_agents * 2)
            Wmask, Imask = imagine_groups
            w1_W = self.hyper_w_1(entities, entity_mask,
                                  attn_mask=Wmask.reshape(bs * max_t,
                                                          ne, ne))
            w1_I = self.hyper_w_1(entities, entity_mask,
                                  attn_mask=Imask.reshape(bs * max_t,
                                                          ne, ne))
            w1 = th.cat([w1_W, w1_I], dim=1)
        else:
            agent_qs = agent_qs.view(-1, 1, self.n_agents)
            # First layer
            w1 = self.hyper_w_1(entities, entity_mask)
        b1 = self.hyper_b_1(entities, entity_mask)
        w1 = w1.view(bs * max_t, -1, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        if self.args.softmax_mixing_weights:
            w1 = F.softmax(w1, dim=-1)
        else:
            w1 = th.abs(w1)

        hidden = self.non_lin(th.bmm(agent_qs, w1) + b1)
        # Second layer
        if self.args.softmax_mixing_weights:
            w_final = F.softmax(self.hyper_w_final(entities, entity_mask), dim=-1)
        else:
            w_final = th.abs(self.hyper_w_final(entities, entity_mask))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(entities, entity_mask).view(-1, 1, 1)

        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot


class LinearFlexQMixer(nn.Module):
    def __init__(self, args):
        super(LinearFlexQMixer, self).__init__()
        self.args = args

        self.n_agents = args.n_agents

        self.embed_dim = args.mixing_embed_dim

        self.hyper_w_1 = AttentionHyperNet(args, mode='alt_vector')
        self.V = AttentionHyperNet(args, mode='scalar')

    def forward(self, agent_qs, inputs, imagine_groups=None, ret_ingroup_prop=False):
        entities, entity_mask = inputs
        bs, max_t, ne, ed = entities.shape

        entities = entities.reshape(bs * max_t, ne, ed)
        entity_mask = entity_mask.reshape(bs * max_t, ne)
        if imagine_groups is not None:
            agent_qs = agent_qs.view(-1, self.n_agents * 2)
            Wmask, Imask = imagine_groups
            w1_W = self.hyper_w_1(entities, entity_mask,
                                  attn_mask=Wmask.reshape(bs * max_t,
                                                          self.n_agents, ne))
            w1_I = self.hyper_w_1(entities, entity_mask,
                                  attn_mask=Imask.reshape(bs * max_t,
                                                          self.n_agents, ne))
            w1 = th.cat([w1_W, w1_I], dim=1)
        else:
            agent_qs = agent_qs.view(-1, self.n_agents)
            # First layer
            w1 = self.hyper_w_1(entities, entity_mask)
        w1 = w1.view(bs * max_t, -1)
        if self.args.softmax_mixing_weights:
            w1 = F.softmax(w1, dim=1)
        else:
            w1 = th.abs(w1)
        v = self.V(entities, entity_mask)

        q_cont = agent_qs * w1
        q_tot = q_cont.sum(dim=1) + v
        # Reshape and return
        q_tot = q_tot.view(bs, -1, 1)
        if ret_ingroup_prop:
            ingroup_w = w1.clone()
            ingroup_w[:, self.n_agents:] = 0  # zero-out out of group weights
            ingroup_prop = (ingroup_w.sum(dim=1)).mean()
            return q_tot, ingroup_prop
        return q_tot


class EntityAttentionLayer(nn.Module):
    def __init__(self, in_dim, embed_dim, out_dim, args):
        super(EntityAttentionLayer, self).__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.n_heads = args.attn_n_heads
        self.n_agents = args.n_agents
        self.args = args

        assert self.embed_dim % self.n_heads == 0, "Embed dim must be divisible by n_heads"
        self.head_dim = self.embed_dim // self.n_heads
        self.register_buffer('scale_factor',
                             th.scalar_tensor(self.head_dim).sqrt())

        self.in_trans = nn.Linear(self.in_dim, self.embed_dim * 3, bias=False)
        self.out_trans = nn.Linear(self.embed_dim, self.out_dim)

    def forward(self, entities, pre_mask=None, post_mask=None, ret_attn_logits=None):
        """
        entities: Entity representations
            shape: batch size, # of entities, embedding dimension
        pre_mask: Which agent-entity pairs are not available (observability and/or padding).
                  Mask out before attention.
            shape: batch_size, # of agents, # of entities
        post_mask: Which agents/entities are not available. Zero out their outputs to
                   prevent gradients from flowing back. Shape of 2nd dim determines
                   whether to compute queries for all entities or just agents.
            shape: batch size, # of agents (or entities)
        ret_attn_logits: whether to return attention logits
            None: do not return
            "max": take max over heads
            "mean": take mean over heads
        Return shape: batch size, # of agents, embedding dimension
        """
        entities_t = entities.transpose(0, 1)
        n_queries = post_mask.shape[1]
        pre_mask = pre_mask[:, :n_queries]
        ne, bs, ed = entities_t.shape
        query, key, value = self.in_trans(entities_t).chunk(3, dim=2)

        query = query[:n_queries]

        query_spl = query.reshape(n_queries, bs * self.n_heads, self.head_dim).transpose(0, 1)
        key_spl = key.reshape(ne, bs * self.n_heads, self.head_dim).permute(1, 2, 0)
        value_spl = value.reshape(ne, bs * self.n_heads, self.head_dim).transpose(0, 1)

        attn_logits = th.bmm(query_spl, key_spl) / self.scale_factor
        if pre_mask is not None:
            pre_mask_rep = pre_mask.repeat_interleave(self.n_heads, dim=0)
            masked_attn_logits = attn_logits.masked_fill(pre_mask_rep[:, :, :ne], -float('Inf'))
        attn_weights = F.softmax(masked_attn_logits, dim=2)
        # some weights might be NaN (if agent is inactive and all entities were masked)
        attn_weights = attn_weights.masked_fill(attn_weights != attn_weights, 0)
        attn_outs = th.bmm(attn_weights, value_spl)
        attn_outs = attn_outs.transpose(
            0, 1).reshape(n_queries, bs, self.embed_dim)
        attn_outs = attn_outs.transpose(0, 1)
        attn_outs = self.out_trans(attn_outs)
        if post_mask is not None:
            attn_outs = attn_outs.masked_fill(post_mask.unsqueeze(2), 0)
        if ret_attn_logits is not None:
            # bs * n_heads, nq, ne
            attn_logits = attn_logits.reshape(bs, self.n_heads,
                                              n_queries, ne)
            if ret_attn_logits == 'max':
                attn_logits = attn_logits.max(dim=1)[0]
            elif ret_attn_logits == 'mean':
                attn_logits = attn_logits.mean(dim=1)
            elif ret_attn_logits == 'norm':
                attn_logits = attn_logits.mean(dim=1)
            return attn_outs, attn_logits
        return attn_outs


class EntityPoolingLayer(nn.Module):
    def __init__(self, in_dim, embed_dim, out_dim, pooling_type, args):
        super(EntityPoolingLayer, self).__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.pooling_type = pooling_type
        self.n_agents = args.n_agents
        self.args = args

        self.in_trans = nn.Linear(self.in_dim, self.embed_dim)
        self.out_trans = nn.Linear(self.embed_dim, self.out_dim)

    def forward(self, entities, pre_mask=None, post_mask=None, ret_attn_logits=None):
        """
        entities: Entity representations
            shape: batch size, # of entities, embedding dimension
        pre_mask: Which agent-entity pairs are not available (observability and/or padding).
                  Mask out before pooling.
            shape: batch_size, # of agents, # of entities
        post_mask: Which agents are not available. Zero out their outputs to
                   prevent gradients from flowing back.
            shape: batch size, # of agents
        ret_attn_logits: not used, here to match attention layer args
        Return shape: batch size, # of agents, embedding dimension
        """
        bs, ne, ed = entities.shape

        ents_trans = self.in_trans(entities)
        n_queries = post_mask.shape[1]
        pre_mask = pre_mask[:, :n_queries]
        # duplicate all entities per agent so we can mask separately
        ents_trans_rep = ents_trans.reshape(bs, 1, ne, ed).repeat(1, self.n_agents, 1, 1)

        if pre_mask is not None:
            ents_trans_rep = ents_trans_rep.masked_fill(pre_mask.unsqueeze(3), 0)

        if self.pooling_type == 'max':
            pool_outs = ents_trans_rep.max(dim=2)[0]
        elif self.pooling_type == 'mean':
            pool_outs = ents_trans_rep.mean(dim=2)

        pool_outs = self.out_trans(pool_outs)

        if post_mask is not None:
            pool_outs = pool_outs.masked_fill(post_mask.unsqueeze(2), 0)

        if ret_attn_logits is not None:
            return pool_outs, None
        return pool_outs
