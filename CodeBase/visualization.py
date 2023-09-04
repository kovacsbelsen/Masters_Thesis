import os
import json
import matplotlib.pyplot as plt
from Full_Project_Code_For_Thesis import Floorplan


def plot_fitness_values(directory):
    fitness_values = []

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename), 'r') as f:
                data = json.load(f)
                floorplan = Floorplan.from_dict(data)
                fitness_values.append(floorplan.fitness)

    # Plot the fitness values
    plt.plot(fitness_values)
    plt.xlabel('Individual')
    plt.ylabel('Fitness')
    plt.show()

plot_fitness_values("D:\\ProgrammingF#\\Master's thesis\\generation0")

plot_fitness_values("D:\\ProgrammingF#\\Master's thesis\\generation1")

plot_fitness_values("D:\\ProgrammingF#\\Master's thesis\\generation2")

plot_fitness_values("D:\\ProgrammingF#\\Master's thesis\\generation3")

plot_fitness_values("D:\\ProgrammingF#\\Master's thesis\\generation4")

plot_fitness_values("D:\\ProgrammingF#\\Master's thesis\\generation5")

plot_fitness_values("D:\\ProgrammingF#\\Master's thesis\\generation6")

plot_fitness_values("D:\\ProgrammingF#\\Master's thesis\\generation7")

plot_fitness_values("D:\\ProgrammingF#\\Master's thesis\\generation8")