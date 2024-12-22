from pathlib import Path

from torch import Tensor

from src.DTO.path_data import PathData

RESULT_DIR = Path(__file__).parents[1] / "results"
if not RESULT_DIR.exists():
    RESULT_DIR.mkdir()


class DataMonitor:
    def __init__(self, number_of_blocks: int, data_name: str) -> None:
        res_dir = RESULT_DIR / data_name
        if not res_dir.exists():
            res_dir.mkdir()
        self.files = [
            open(res_dir / f"wta_{i}_activity.txt", "w")
            for i in range(number_of_blocks)
        ]
        self.res = open(res_dir / "result.txt", "w")

    def save_wta_inference(self, data: Tensor, block_number: int) -> None:
        self.files[block_number].write(
            ", ".join(map(str, data[0, :, -1].tolist())) + "\n"
        )

    def save_model_inference(self, path_data: PathData) -> None:
        self.res.write(str(f"{path_data.path} {path_data.distance}\n"))

    def close(self) -> None:
        for f in self.files:
            f.close()
        self.res.close()
