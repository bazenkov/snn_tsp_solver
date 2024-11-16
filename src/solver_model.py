import torch
from torch.nn import ModuleList
from tqdm import tqdm

from .commutator import BlockCommutator
from .noise_gen import NoiseGenerator
from .neurons import PossibleNeuronModels


class TSPModel:
    def __init__(
        self,
        numb_of_cities: int,
        feedback_coefficient: float,
        weights: torch.Tensor,
        neuron_model: PossibleNeuronModels,
        temp: float,
        data: torch.Tensor,
    ) -> None:
        self.temperature = temp
        self.time = None
        self.numb_of_cities = numb_of_cities
        self.feedback_coefficient = feedback_coefficient
        self.weights = weights
        self.neuron_model = neuron_model
        self.data = torch.Tensor()
        self.blocks = self.create_wta_blocks()
        self.start_spike = self.create_start_spike()
        self.path = [-1] * self.numb_of_cities
        self.distance = -1
        self.max_distance = -1
        self.commutator = BlockCommutator(
            inference_shape=self.start_spike.shape,
            number_of_cities=self.numb_of_cities,
        )
        self.noise_generator = None
        self.data = data

    def create_wta_blocks(self) -> ModuleList:
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

    def create_start_spike(self) -> torch.Tensor:
        start_spike = torch.zeros(1, self.numb_of_cities, 2)
        start_spike[:, 0, :] += 1.0
        return start_spike

    def forward(self, input_spike: torch.Tensor, t: int) -> torch.Tensor:
        for i, block in enumerate(self.blocks):
            block.bias = self.feedback_coefficient * self.commutator.get_bias(port=i)
            input_spike = block(input_spike)

            self.path[i] = self.current_travel_node(input_spike)
            self.commutator.set_bias(port=i, bias=input_spike)

            input_spike += self.noise_generator.gen_noise(
                self.distance, self.max_distance, t
            )
        return input_spike

    @staticmethod
    def current_travel_node(block_inference: torch.Tensor) -> int:
        activ_neurons = block_inference.nonzero()
        if len(activ_neurons) == 1:
            return activ_neurons[0, 1].item()
        return -1

    def current_distance(self) -> int:
        if len(set(self.path)) != len(self.path) or -1 in self.path:
            return -1
        x, y = self.path[0], self.path[1]
        distance = self.data[x, y]
        for i in range(2, len(self.path), 2):
            d_y = self.path[i]
            distance += self.data[x, d_y]
            if i + 1 < len(self.data):
                break
            d_x = self.path[i + 1]
            distance += self.data[d_x, d_y]
            x = d_x
        self.max_distance = max(distance, self.max_distance)
        return distance

    def solve(self, time: int, output_file: str) -> None:
        self.time = time
        input_spike = self.start_spike.clone()

        self.noise_generator = NoiseGenerator(
            inference_shape=self.start_spike.shape,
            time_simulation=time,
            temperature=self.temperature,
        )

        with open(output_file, "w") as file:
            for t in tqdm(range(time)):
                input_spike = self.forward(input_spike, t)
                self.distance = self.current_distance()
                file.write(f"{self.path} {self.distance}\n")

    def clear(self):
        self.__init__(
            numb_of_cities=self.numb_of_cities,
            feedback_coefficient=self.feedback_coefficient,
            weights=self.weights,
            neuron_model=self.neuron_model,
            temp=self.temperature,
            data=self.data,
        )
