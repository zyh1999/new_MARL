REGISTRY = {}

from .basic_controller import BasicMAC
from .wqmix_controller import CentralBasicMAC
from .token_controller import TokenMAC
from .token_latent_controller import TokenLatentMAC
from .entity_controller import EntityMAC
from .entity_refil_controller import EntityRefilMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["wqmix_mac"] = CentralBasicMAC
REGISTRY["token_mac"] = TokenMAC
REGISTRY["token_latent_mac"] = TokenLatentMAC
REGISTRY["entity_mac"] = EntityMAC
REGISTRY["entity_refil_mac"] = EntityRefilMAC


