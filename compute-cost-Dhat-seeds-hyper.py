from EncoderDecoder.utils import loadData
from PCAfold import compute_normalized_variance, normalized_variance_derivative, cost_function_normalized_variance_derivative, plot_normalized_variance_derivative
import numpy as np
import matplotlib.pyplot as plt
import sys
from itertools import product
import logging
import torch

logging.disable(logging.CRITICAL)

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

def compute_avg(costs):
    n = len(costs)
    sum = np.sum(costs**2)
    return 1/n*np.sqrt(sum)

nbr_seeds = 6

general_dataset_type = "Malik"
dataset_type = "CFLF-2022"

short_id = "1d2PV0to1"
date = "01Aug2026-hour_17h05"
epochs = 100000

learning_rates = [0.025]
optimizers = ["RMSprop"]
lists_species_output = [
    ("major", ["CH4", "CO", "O2", "CO2", "H2O", "N2"])
]
seeds = list(range(nbr_seeds))

experiment_configs = []

for lr_i, opt_i, (species_tag, species_i), seed_i in product(
    learning_rates,
    optimizers,
    lists_species_output,
    seeds):

    config = {
        "lr": lr_i,
        "optimizer": opt_i,
        "output_species": species_i,
        "species_tag": species_tag,
        "seed": seed_i,
    }
    experiment_configs.append(config)

print(f"Total number of runs: {len(experiment_configs)}")

list_avg_cost = []

for idxConfig, config in enumerate(experiment_configs):

    optimizer_name = config["optimizer"]
    lr = config["lr"]
    species_tag = config["species_tag"]
    my_seed = config["seed"]

    training_nbr = f"{short_id}_{optimizer_name}_{int(lr*10000)}_{species_tag}_s{my_seed}"
    filename = f"Tr{training_nbr}-AE-date_{date}_{general_dataset_type}-{dataset_type}"

    path_data = "data-files/"
    filename_metadata = filename + "_metadata.pkl"
    filename_species_names = f"{general_dataset_type}-state-space-names-{dataset_type}.csv"
    path_metadata = "metadata/"

    penalty_function = 'log-sigma-over-peak'
    start_bw = -6
    end_bw = 2
    nbr_points_bw = 100
    bandwidth_values = np.logspace(start_bw, end_bw, nbr_points_bw)
    power = 4
    vertical_shift = 1

    loader = loadData(filename)
    depvar_names = loader.getListQoIs()
    model = loader.loadModel(device = device)
    id = loader.metadata["training_id"]

    #get the input (PV and f) and the output (interested Yi, T and source terms) data
    PV_f, output = loader.getInputOutputAnalysis(path_data, dataset_type) #for PV_f reshape to be (5200,1) instead of (52000)

    #scale every column of the PV_f tensor between 0 and 1
    min_vals = np.min(PV_f, axis=0, keepdims=True)
    max_vals = np.max(PV_f, axis=0, keepdims=True)
    PV_f_scaled = (PV_f - min_vals) / (max_vals - min_vals)

    indepVars = PV_f_scaled
    depVars = output

    variance_data = compute_normalized_variance(indepVars,
                                                    depVars,
                                                    depvar_names=depvar_names,
                                                    bandwidth_values=bandwidth_values)
    np.save(f"data-files/costs/variance_{id}-bw_{start_bw}_{end_bw}_{nbr_points_bw}-dataset_{dataset_type}.npy", variance_data)

    costs = cost_function_normalized_variance_derivative(variance_data,
                                                        penalty_function=penalty_function,
                                                        power=power,
                                                        vertical_shift=vertical_shift,
                                                        norm=None)
    np.save(f"data-files/costs/costs_{id}-bw_{start_bw}_{end_bw}_{nbr_points_bw}-p_{power}-ver_sh_{vertical_shift}-dataset_{dataset_type}.npy", costs)

    (derivative, bandwidth_values, max_derivative) = normalized_variance_derivative(variance_data)

    plt = plot_normalized_variance_derivative(variance_data)
    plt.savefig(f"data-files/costs/plot_Dhat_{id}-bw_{start_bw}_{end_bw}_{nbr_points_bw}-p_{power}-ver_sh_{vertical_shift}-dataset_{dataset_type}.png")

    list_avg_cost.append(compute_avg(np.array(costs)))
    print(f"{filename} done.")

print()
print("Computation complete")
print()

for cost in list_avg_cost:
    print(f"{np.round(cost,2)}")