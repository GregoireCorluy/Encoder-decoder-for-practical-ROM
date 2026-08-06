import logging
import pandas as pd
from PCAfold import KernelDensity
import numpy as np
logging.disable(logging.CRITICAL)

path = "data-files/"
name_state_space = "Malik-state-space-CFLF-2022.csv"

species_conditioning = "CO2"

species = ["CH4", "CO", "O2", "CO2", "H2O", "N2"]

data_state_space = pd.read_csv(path + name_state_space)
variable_conditioning = data_state_space["CO2"].to_numpy()

data_state_space_major = data_state_space[species]
X = data_state_space_major.to_numpy()

# center and scale
mean = np.mean(X, axis = 0)
std = np.std(X, axis = 0)
X_scaled = (X - mean)/np.sqrt(std)
variable_conditioning = (variable_conditioning - np.mean(variable_conditioning))/np.sqrt(np.std(variable_conditioning))

kerneld = KernelDensity(X_scaled, variable_conditioning)

weights = kerneld.weights

np.save("data-files/extraData/kernelDensity/2026-08-06-kernelDensity-Malik-CFLF-2022-condition-CO2-dataCenteredScaled", weights)