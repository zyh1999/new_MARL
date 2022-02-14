REGISTRY = {}

from .rnn_agent import RNNAgent
REGISTRY["rnn"] = RNNAgent
'''
from .wqmix_agent import CentralRNNAgent
REGISTRY["wqmix"] = CentralRNNAgent

from .token_updet_agent import UPDeT
REGISTRY['token_updet'] = UPDeT

from .token_dyan_agent import DyAN
REGISTRY['token_dyan'] = DyAN

from .token_trans_agent import TokenTransAgent
REGISTRY['token_trans'] = TokenTransAgent

from .token_attn_agent import TokenAttnAgent
REGISTRY["token_attn"] = TokenAttnAgent

from .token_branch_attn_agent import TokenBranchAttnAgent
REGISTRY["token_branch_attn"] = TokenBranchAttnAgent

from .token_wise_trans_agent import TokenWiseTransAgent
REGISTRY['token_wise_trans'] = TokenWiseTransAgent

from .token_wise_attn_agent import TokenWiseAttnAgent
REGISTRY["token_wise_attn"] = TokenWiseAttnAgent

from .token_wise_branch_attn_agent import TokenWiseBranchAttnAgent
REGISTRY["token_wise_branch_attn"] = TokenWiseBranchAttnAgent

from .token_latent_pattern_wise_trans_agent import TokenLatentPatternWiseTransAgent
REGISTRY["token_latent_pattern_wise_trans"] = TokenLatentPatternWiseTransAgent

from .entity_attn_agent import EntityAttnAgent
REGISTRY["entity_attn"] = EntityAttnAgent

from .entity_trans_agent import EntityTransAgent
REGISTRY["entity_trans"] = EntityTransAgent

from .entity_sparse_attn_agent import EntitySparseAttnAgent
REGISTRY["entity_sparse_attn"] = EntitySparseAttnAgent

from .entity_sparse_trans_agent import EntitySparseTransAgent
REGISTRY["entity_sparse_trans"] = EntitySparseTransAgent

from .entity_spotlight_trans_agent import EntitySpotlightTransAgent
REGISTRY["entity_spotlight_trans"] = EntitySpotlightTransAgent

from .entity_pattern_trans_agent import EntityPatternTransAgent
REGISTRY["entity_pattern_trans"] = EntityPatternTransAgent

from .entity_inner_pattern_trans_agent import EntityInnerPatternTransAgent
REGISTRY["entity_inner_pattern_trans"] = EntityInnerPatternTransAgent

from .entity_latent_inner_pattern_trans_agent import EntityLatentInnerPatternTransAgent
REGISTRY["entity_latent_inner_pattern_trans"] = EntityLatentInnerPatternTransAgent

from .entity_refil_agent import ImagineEntityAttentionRNNAgent
REGISTRY["entity_refil"] = ImagineEntityAttentionRNNAgent
'''