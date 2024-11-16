import torch
from torch import Size


class BlockCommutator:
    def __init__(self, inference_shape: Size, number_of_cities: int) -> None:
        self.data = torch.zeros(number_of_cities, *inference_shape)

    def set_bias(self, port: int, bias: torch.Tensor) -> None:
        self.data[port] = bias

    def get_bias(self, port: int) -> torch.Tensor:
        res = torch.sum(self.data, dim=0) - self.data[port]
        return res
