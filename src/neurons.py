from dataclasses import dataclass
from enum import Enum
from typing import Dict, Type

from lava.lib.dl.slayer.block import base, alif, cuba


@dataclass
class NeuronModel:
    neuron_config: Dict[str, float]
    block_model: Type[base.AbstractKWTA]


class PossibleNeuronModels(Enum):
    ALIF = NeuronModel(
        neuron_config={
            "threshold": 0.1,
            "threshold_step": 0.01,
            "current_decay": 1,
            "voltage_decay": 0.1,
            "threshold_decay": 0.1,
            "refractory_decay": 0.1,
        },
        block_model=alif.KWTA,
    )

    CUBA = NeuronModel(
        neuron_config={
            "threshold": 0.1,
            "current_decay": 1,
            "voltage_decay": 0.1,
        },
        block_model=cuba.KWTA,
    )
