import numpy as np
import torch

from .neurons import PossibleNeuronModels
from .solver_model import TSPModel


class TSPSolver:
    def __init__(
        self,
        input_file: str,
        output_file: str,
        neuron_model: PossibleNeuronModels,
        feedback_coefficient: float,
        temp: float,
    ) -> None:
        self.neuron_model = neuron_model
        self.temperature = temp
        self.output_file = output_file
        self.data = self.__load_input_data(input_file)
        self.numb_of_cities = len(self.data)
        self.solver_model = self.__load_solver_model(neuron_model, feedback_coefficient)

    @staticmethod
    def __load_input_data(input_file: str) -> np.ndarray:
        with open(input_file, "r") as f:
            data = [[float(num) for num in line.split()] for line in f]
        return np.array(data)

    def __load_solver_model(
        self, neuron_model: PossibleNeuronModels, feedback_coefficient: float
    ) -> TSPModel:
        w_offset = 0.1
        w_scale = 0.1

        torch_path_data = torch.tensor(self.data, dtype=torch.float32) / np.max(
            self.data
        )
        torch_path_data = (
            w_offset
            + (1 - torch_path_data) * w_scale
            - 2 * w_offset * torch.eye(self.numb_of_cities)
        )

        model = TSPModel(
            numb_of_cities=self.numb_of_cities,
            feedback_coefficient=feedback_coefficient,
            weights=torch_path_data,
            data=self.data,
            neuron_model=neuron_model,
            temp=self.temperature,
        )

        return model

    def solve(self, time: int, epoch: int = 1):
        for i in range(epoch):
            print(f"epoch {i + 1}")
            self.solver_model.solve(time=time, output_file=self.output_file)
            self.solver_model.clear()
