import torch
import lava.lib.dl.slayer.block.alif as alif
import lava.lib.dl.slayer.block.cuba as cuba
import numpy as np
from tqdm import tqdm


class BlockCommutator:
    def __init__(self, start_spike, ports_num, temp=0.4):
        self.start_spike = start_spike
        self.data = torch.zeros(ports_num, *start_spike.shape)
        self.temp = temp
        self.distance_factor = None
        self.max_distance_factor = None

    def set_bias(self, bias, port):
        self.data[port] = bias

    def get_bias(self, port, time, feedback_cof):
        res = torch.sum(self.data, dim=0) - self.data[port] + self.start_spike
        if port == self.start_spike.shape[1] - 1:
            res -= self.start_spike
        rand = np.exp(-0.00002 * time) * (torch.rand_like(res) * 2 - 1)
        if self.distance_factor:
            rand = rand * self.distance_factor / self.max_distance_factor
        return feedback_cof * res + self.temp * rand


def path_calc(data, path_data):
    path = []
    for wta_block in data:
        if len(pos := wta_block.nonzero().squeeze(dim=-1)) == 1:
            path.append(pos.item())
        else:
            return None, None

    x, y = path[0], path[1]
    distance = path_data[x, y]
    for i in range(2, len(path), 2):
        d_y = path[i]
        distance += path_data[x, d_y]
        if i + 1 < len(path_data):
            break
        d_x = path[i + 1]
        distance += path_data[d_x, d_y]
        x = d_x

    distance += path_data[path[0], path[-1]]

    return path, distance


class TSPSolver(torch.nn.Module):
    def __init__(self, numb_of_cities, path_data):
        super(TSPSolver, self).__init__()

        alif_neuron_params = {
            "threshold": 0.1,
            "threshold_step": 0.01,
            "current_decay": 1,
            "voltage_decay": 0.1,
            "threshold_decay": 0.1,
            "refractory_decay": 0.1,
        }

        cuba_neuron_params = {
            "threshold": 0.1,
            "current_decay": 1,
            "voltage_decay": 0.1,
        }

        self.n = numb_of_cities
        self.weights = path_data
        self.feedback_cof = -1.5
        self.data = torch.Tensor()

        # создается N-1 WTA блоков
        self.blocks = torch.nn.ModuleList(
            [
                cuba.KWTA(
                    cuba_neuron_params,
                    numb_of_cities,
                    numb_of_cities,
                    1,
                    delay_shift=False,
                    self_excitation=0.5,
                )
                for _ in range(numb_of_cities)
            ]
        )

        custom_weights_param = torch.nn.Parameter(
            path_data.reshape((numb_of_cities, numb_of_cities, 1, 1, 1))
        )
        for block in self.blocks:
            block.synapse.weight = custom_weights_param

    def solve(self, time):
        self.t = time
        start_input_spike = torch.zeros(1, self.n, 2)
        start_input_spike[:, 0, :] += 1.0
        self.data = torch.Tensor()
        self.comutator = BlockCommutator(start_input_spike, self.n)
        self.path = [[-1] * self.n]
        self.distance = [-1]

        for t in tqdm(range(time)):
            one_time_data = start_input_spike[:, :, -1]
            input_spike = start_input_spike.clone()
            for i, block in enumerate(self.blocks):
                block.bias = self.comutator.get_bias(
                    port=i, time=t, feedback_cof=self.feedback_cof
                )
                input_spike = block(input_spike)
                self.comutator.set_bias(input_spike, port=i)
                one_time_data = torch.cat((one_time_data, input_spike[:, :, -1]), 0)

            # self.data = torch.cat((self.data, one_time_data), dim=-1)
            path, distance = path_calc(one_time_data, distance_matrix)

            if path:
                self.path.append(path)
                self.distance.append(distance)
                self.comutator.distance_factor = distance
                self.comutator.max_distance_factor = self.distance[-1]

    def save_data(self, epoch):
        name = f"solves/26_tsp_solve_{epoch + 1}.txt"
        with open(name, "w") as file:
            for k in range(1, len(self.distance)):
                file.write(f"{self.distance[k]}  {self.path[k]}\n")


if __name__ == "__main__":
    with open("example4.txt", "r") as f:
        data = [[int(num) for num in line.split()] for line in f]

    numb_of_cities = 4
    path_data = np.array(data)

    w_offset = 0.1
    w_scale = 0.1

    torch_path_data = torch.tensor(path_data, dtype=torch.float32) / np.max(path_data)
    torch_path_data = (
        w_offset
        + (1 - torch_path_data) * w_scale
        - 2 * w_offset * torch.eye(numb_of_cities)
    )
    distance_matrix = np.array(path_data)

    for i in range(1):
        print(f"epoch {i + 1}")
        net = TSPSolver(numb_of_cities=numb_of_cities, path_data=torch_path_data)
        net.solve(time=1000)
        net.save_data(i)

# 937
