from .entity_controller import EntityMAC
import torch as th


# This multi-agent controller shares parameters between agents
# takes entities + observation masks as input
class EntityRefilMAC(EntityMAC):
    def __init__(self, scheme, groups, args):
        super(EntityRefilMAC, self).__init__(scheme, groups, args)

    def forward(self, ep_batch, t, test_mode=False, imagine=False):
        if t is None:
            t = slice(0, ep_batch["entities"].shape[1])
            single_step = False
        else:
            t = slice(t, t + 1)
            single_step = True

        agent_inputs = self._build_inputs(ep_batch, t)
        if imagine:
            agent_outs, self.hidden_states, groups = self.agent(agent_inputs, self.hidden_states, imagine)
        else:
            agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        if self.agent_output_type == "pi_logits":
            assert False, "unsupported agent_output_type"

        if single_step:
            return agent_outs.squeeze(1)
        if imagine:
            return agent_outs, groups
        return agent_outs

