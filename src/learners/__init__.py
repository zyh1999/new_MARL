from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .qplex_learner import DMAQ_qattenLearner as QPlexLearner
'''
from .wqmix_learner import MAXQLearner as WQMixLearner
from .token_q_learner import QLearner as TokenQLearner
from .entity_q_learner import QLearner as EntityQLearner
from .entity_disentangle_q_learner import QLearner as EntityDisentangleQLearner
from .entity_refil_q_learner import QLearner as EntityRefilQLearner
from .token_disentangle_q_learner import QLearner as TokenDisentangleQLearner
'''
REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["qplex_learner"] = QPlexLearner
'''
REGISTRY["wqmix_learner"] = WQMixLearner
REGISTRY["token_q_learner"] = TokenQLearner
REGISTRY["entity_q_learner"] = EntityQLearner
REGISTRY["entity_disentangle_q_learner"] = EntityDisentangleQLearner
REGISTRY["entity_refil_q_learner"] = EntityRefilQLearner
REGISTRY["token_disentangle_q_learner"] = TokenDisentangleQLearner
'''