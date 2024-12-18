from numpy import ndarray
from torch import Tensor


def current_travel_node(block_inference: Tensor) -> int:
    activ_neurons = block_inference.nonzero()
    if len(activ_neurons) == 1:
        return activ_neurons[0, 1].item()
    return -1


def current_distance(path: list[int], data: ndarray) -> int:
    if len(set(path)) != len(path) or -1 in path:
        return -1
    x, y = path[0], path[1]
    distance = data[x, y]
    for i in range(2, len(path), 2):
        d_y = path[i]
        distance += data[x, d_y]
        if i + 1 < len(data):
            break
        d_x = path[i + 1]
        distance += data[d_x, d_y]
        x = d_x
    return distance
