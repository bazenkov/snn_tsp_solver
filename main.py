from src.neurons import PossibleNeuronModels
from src.solver import TSPSolver

if __name__ == "__main__":
    solver = TSPSolver(
        input_file="recourses/five_d.txt",
        output_file="results/five_d_res.txt",
        neuron_model=PossibleNeuronModels.CUBA,
        feedback_coefficient=-1.5,
        temp=0.4,
    )
    solver.solve(time=5000, epoch=1)
