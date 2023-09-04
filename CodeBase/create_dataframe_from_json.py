import os
import json
from Full_Project_Code_For_Thesis import tree_to_string
from Full_Project_Code_For_Thesis import Floorplan
import pandas as pd


def create_dataframe_from_json(directory):
    
    # List to store data dictionaries
    data_list = []

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename), 'r') as f:
                data = json.load(f)
                floorplan = Floorplan.from_dict(data)
                
                # Create a dictionary with all the required values
                data_dict = {
                    'filename': filename,
                    'num_rooms': floorplan.num_rooms,
                    #'slicing_tree_root': floorplan.slicing_tree_root,
                    #'floorplan_instance': floorplan,
                    'fitness': floorplan.fitness,
                    #'rooms': floorplan.rooms,
                    #'doors_list': floorplan.doors_list,
                    'tree_string': tree_to_string(floorplan.slicing_tree_root),  # assuming tree_to_string is a function that takes a slicing_tree_root as input and returns a string
                    'agents_paths': floorplan.agents_paths,
                    'agents_activites': floorplan.agents_activites,
                    'neighbouring_rooms': floorplan.neighbouring_rooms,
                    'mutated': floorplan.mutated,
                    'crossover': floorplan.crossover,
                    'parent1': floorplan.parent1,
                    'parent2': floorplan.parent2,
                    'mutation_probability': floorplan.mutation_probability,
                    'agents_have_bedrooms"': floorplan.agents_have_bedrooms
                }

                # Append the dictionary to the data list
                data_list.append(data_dict)

    # Create a DataFrame from the data list
    df = pd.DataFrame(data_list)
    
    return df