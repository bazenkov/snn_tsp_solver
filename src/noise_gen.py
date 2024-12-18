import numpy as np
import torch
from torch import Size


class NoiseGenerator:
    def __init__(
        self,
        inference_shape: Size,
        time_simulation: int,
        temperature: float,
    ) -> None:
        self.inference_shape = inference_shape
        self.time_simulation = time_simulation
        self.temp = temperature

    def time_noise(self, t: int) -> torch.Tensor:
        return np.exp(-1 / self.time_simulation / 2 * t) * (
            torch.rand(self.inference_shape) * 2 - 1
        )

    @staticmethod
    def path_noise(distance: int, max_distance: int) -> float:
        if distance != -1 and max_distance != 0:
            return distance / max_distance
        return 1

    def gen_noise(self, distance: int, max_distance: int, t: int) -> torch.Tensor:
        return self.temp * self.path_noise(distance, max_distance) * self.time_noise(t)
