import numpy as np
import torch
from torch.nn import ModuleList
from tqdm import tqdm

from .commutator import BlockCommutator
from .data_monitor import DataMonitor
from .noise_gen import NoiseGenerator
from .neurons import PossibleNeuronModels
from .utils import current_travel_node, current_distance


class TSPModel:
    def __init__(
        self,
        numb_of_cities: int,
        feedback_coefficient: float,
        weights: torch.Tensor,
        neuron_model: PossibleNeuronModels,
        temp: float,
        data: np.ndarray,
        data_name: str,
    ) -> None:
        self.temperature = temp
        self.time = None
        self.numb_of_cities = numb_of_cities
        self.feedback_coefficient = feedback_coefficient
        self.weights = weights
        self.neuron_model = neuron_model
        self.data = torch.Tensor()
        self.blocks = self._create_wta_blocks()
        self.start_spike = self._create_start_spike()
        self.path = [-1] * self.numb_of_cities
        self.distance = -1
        self.max_distance = -1
        self.commutator = BlockCommutator(
            inference_shape=self.start_spike.shape,
            number_of_cities=self.numb_of_cities,
        )
        self.noise_generator = None
        self.data = data
        self.data_name = data_name
        self.monitor = DataMonitor(
            number_of_blocks=numb_of_cities, data_name=self.data_name
        )

    def _create_wta_blocks(self) -> ModuleList:
        block_model = self.neuron_model.value.block_model
        neurons_config = self.neuron_model.value.neuron_config
        blocks = ModuleList(
            [
                block_model(
                    neuron_params=neurons_config,
                    in_neurons=self.numb_of_cities,
                    out_neurons=self.numb_of_cities,
                    num_winners=1,
                    delay_shift=False,
                    self_excitation=0.5,
                )
                for _ in range(self.numb_of_cities)
            ]
        )

        custom_weights_param = torch.nn.Parameter(
            self.weights.reshape(
                shape=(self.numb_of_cities, self.numb_of_cities, 1, 1, 1)
            )
        )

        for block in blocks:
            block.synapse.weight = custom_weights_param

        return blocks

    def _create_start_spike(self) -> torch.Tensor:
        start_spike = torch.zeros(1, self.numb_of_cities, 2)
        start_spike[:, 0, :] += 1.0
        return start_spike

    def forward(self, input_spike: torch.Tensor, t: int) -> torch.Tensor:
        for i, block in enumerate(self.blocks):
            block.bias = self.feedback_coefficient * self.commutator.get_bias(port=i)
            input_spike = block(input_spike)
            self.monitor.save_wta_inference(input_spike, i)

            self.path[i] = current_travel_node(input_spike)
            self.commutator.set_bias(port=i, bias=input_spike)

            input_spike += self.noise_generator.gen_noise(
                self.distance, self.max_distance, t
            )
        return input_spike

    def solve(self, time: int) -> None:
        self.time = time
        input_spike = self.start_spike.clone()

        self.noise_generator = NoiseGenerator(
            inference_shape=self.start_spike.shape,
            time_simulation=time,
            temperature=self.temperature,
        )

        for t in tqdm(range(time)):
            input_spike = self.forward(input_spike, t)
            self.distance = max(
                self.distance, current_distance(path=self.path, data=self.data)
            )
            self.monitor.save_model_inference(self.path, self.distance)

    def clear(self):
        self.__init__(
            numb_of_cities=self.numb_of_cities,
            feedback_coefficient=self.feedback_coefficient,
            weights=self.weights,
            neuron_model=self.neuron_model,
            temp=self.temperature,
            data=self.data,
            data_name=self.data_name,
        )
