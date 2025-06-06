import sys, os
sys.path.insert(0, './src')

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.font_manager import FontProperties
legend_font = FontProperties(family='Arial', style='normal', size=9)
import seaborn as sns
from sklearn.metrics import r2_score

from gnn_eads.functions import structure_to_graph
from gnn_eads.nets import PreTrainedModel

MODEL_NAME = "GNN_to_predict"
MODEL_PATH = "../models/{}".format(MODEL_NAME)  
model = PreTrainedModel(MODEL_PATH)
print(model)

DATA_PATH = "/work/home/acdsqiq3o7/user/jhq/GNN/data_re/Pred_dataset/data_to_pre"  # Path to the dataset

# Read the data
df = pd.read_csv(os.path.join(DATA_PATH, "energies.dat"), sep=" ", header=None)

df.columns = ["structure"]

graphs, energies_GNN = [], []
for row, system in df.iterrows():
    try:
        file_path = os.path.join(DATA_PATH, "contcars", "{}.contcar".format(df["structure"][row]))
        graphs.append(structure_to_graph(file_path,
                                       model.g_tol,    
                                       model.g_sf,
                                       model.g_metal_2nn))
        # print(graphs)
        energies_GNN.append(model.evaluate(graphs[-1]))
        # print(energies_GNN)
        print("Done with {}".format(df["structure"][row]))
    except Exception as e:
        print("Error in {}: {}".format(df["structure"][row], str(e)))
        graphs.append(None)
        energies_GNN.append(None)
# Add graphs to dataframe
atom_num = [None if graph is None else len(graph.species_list) for graph in graphs]


df["graph"] = graphs
df["atom_num"] = atom_num
df["energies_GNN"] = energies_GNN
# # Delete rows with None graphs
df = df[df["graph"].notna()]

# Save the dataframe to a CSV file
df.to_csv("/work/home/acdsqiq3o7/user/jhq/GNN/predictions.csv")
