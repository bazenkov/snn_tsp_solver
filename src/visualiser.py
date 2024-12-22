import numpy as np
from matplotlib import pyplot as plt

from src.data_monitor import RESULT_DIR


class Visualiser:
    def __init__(self, data_name: str):
        self.data_path = RESULT_DIR / data_name

    def show_wta_dynamic(self, block_number: int) -> None:
        spikes = np.genfromtxt(
            self.data_path / f"wta_{block_number}_activity.txt",
            delimiter=",",
        )
        spikes = spikes.T
        offsets = list(range(1, len(spikes) + 1))

        data = [[j for j, item in enumerate(line) if item] for line in spikes]

        plt.eventplot(positions=data, lineoffsets=offsets, linelengths=0.5)
        plt.title(f"WTA block №{block_number + 1}")
        plt.yticks(offsets)
        plt.xlabel("Time steps")
        plt.ylabel("Neurons")

        plt.show()
