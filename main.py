from src.neurons import PossibleNeuronModels
from src.solver import TSPSolver
from src.visualiser import Visualiser

if __name__ == "__main__":
    solver = TSPSolver(
        input_file="recourses/five_d.txt",
        neuron_model=PossibleNeuronModels.CUBA,
        feedback_coefficient=-1.5,
        temp=0.4,
    )
    solver.solve(time=1500, epoch=1)
    del solver
    visualiser = Visualiser(data_name="five_d")
    for i in range(5):
        visualiser.show_wta_dynamic(i)
