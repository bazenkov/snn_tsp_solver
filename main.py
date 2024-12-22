from src.DTO.neurons import PossibleNeuronModels
from src.solver import TSPSolver
from src.visualiser import Visualiser

if __name__ == "__main__":
    solver = TSPSolver(
        input_file="recourses/five_d.txt",
        neuron_model=PossibleNeuronModels.CUBA,
        feedback_coefficient=-2,
        temp=0.6,
    )
    solver.solve(time=500)

    visualiser = Visualiser(data_name="five_d")
    for i in range(5):
        visualiser.show_wta_dynamic(i)
