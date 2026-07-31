import logging
import torch
logging.disable(logging.CRITICAL)

from PCAfold import compute_normalized_variance, normalized_variance_derivative, cost_function_normalized_variance_derivative, plot_normalized_variance_derivative, PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

dataset_type = "autoignition"

penalty_function = 'log-sigma-over-peak'
start_bw = -6
end_bw = 2
nbr_points_bw = 100
bandwidth_values = np.logspace(start_bw, end_bw, nbr_points_bw)
power = 4
vertical_shift = 1
sample_norm_var = False
sample_norm_range = False

#Compute the cost for the PV of Xu via a model where the encoder is modified

def compute_avg(costs):
    n = len(costs)
    sum = np.sum(costs**2)
    return 1/n*np.sqrt(sum)

path = "data-files/"
name_T = "Malik-T-CFLF-2022.csv"
name_state_space = "Malik-state-space-CFLF-2022.csv"
name_state_space_source = "Malik-state-space_source-CFLF-2022.csv"

data_T = pd.read_csv(path + name_T)
data_state_space = pd.read_csv(path + name_state_space)
data_state_space_source = pd.read_csv(path + name_state_space_source)

species_selection_name = "major"
species = ["CH4", "CO", "O2", "CO2", "H2O", "N2"]
scaling = "pareto"

data_state_space = data_state_space[species]
state_space_numpy = data_state_space.to_numpy()

pca_state_space = PCA(
        state_space_numpy,
        scaling=scaling,
        n_components=2,
        use_eigendec=True,
        nocenter=False
    )

PCA_param = pca_state_space.transform(state_space_numpy)


depvar_names = species + ["T", "PC1", "PC2"]

output = data_state_space[species]
output = pd.concat([output, data_T], axis=0, ignore_index=True).to_numpy()

dataSourceMajor = data_state_space_source[species].to_numpy()
PCsource = pca_state_space.transform(dataSourceMajor)

output = np.concatenate([output, PCsource[:, :2]], axis=1)

#scale every column of the PCA_param array between 0 and 1
min_vals = PCA_param.min(axis=0, keepdims=True)
max_vals = PCA_param.max(axis=0, keepdims=True)
PCA_param_scaled = (PCA_param - min_vals) / (max_vals - min_vals)

min_vals_output = output.min(axis=0, keepdims=True)
max_vals_output = output.max(axis=0, keepdims=True)
output_scaled = (output - min_vals_output) / (max_vals_output - min_vals_output)

indepVars = PCA_param_scaled
depVars = output_scaled

print("Compute the variance")
variance_data = compute_normalized_variance(indepVars,
                                                depVars,
                                                depvar_names=depvar_names,
                                                bandwidth_values=bandwidth_values,
                                                compute_sample_norm_range=sample_norm_range,
                                                compute_sample_norm_var=sample_norm_var)
print("Computing the variance is fininshed")
np.save(f"data-files/costs/variance_PCA_{scaling}_{species_selection_name}-bw_{start_bw}_{end_bw}_{nbr_points_bw}-dataset_{dataset_type}{'-sample-norm-var' if sample_norm_var else ''}{'-sample-norm-range' if sample_norm_range else ''}.npy", variance_data)

print("Compute the costs")
costs = cost_function_normalized_variance_derivative(variance_data,
                                                    penalty_function=penalty_function,
                                                    power=power,
                                                    vertical_shift=vertical_shift,
                                                    norm=None)
np.save(f"data-files/costs/costs_PCA_{scaling}_{species_selection_name}-bw_{start_bw}_{end_bw}_{nbr_points_bw}-p_{power}-ver_sh_{vertical_shift}-dataset_{dataset_type}{'-sample-norm-var' if sample_norm_var else ''}{'-sample-norm-range' if sample_norm_range else ''}.npy", costs)

print("Compute the derivatives")
(derivative, bandwidth_values, max_derivative) = normalized_variance_derivative(variance_data)

print("Plot the derivatives")
plt = plot_normalized_variance_derivative(variance_data)
plt.savefig(f"data-files/costs/plot_Dhat_PCA_{scaling}_{species_selection_name}-bw_{start_bw}_{end_bw}_{nbr_points_bw}-p_{power}-ver_sh_{vertical_shift}-dataset_{dataset_type}{'-sample-norm-var' if sample_norm_var else ''}{'-sample-norm-range' if sample_norm_range else ''}.png")
#plt.show()

print(f"Cost of PCA_{scaling} for dataset {dataset_type}")
print(f"{np.round(compute_avg(np.array(costs)),3)}")