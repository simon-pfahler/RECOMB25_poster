import os

import fastmhn
import numpy as np
from mhn.utilities import sample_artificial_data

from parameters import *

np.random.seed(42)

# loop over datasets
for nr_dataset in range(nr_datasets):
    print(f"Logging info: Generating dataset {nr_dataset}")
    # generate ground truth theta matrix
    theta_GT = fastmhn.utility.generate_theta(
        d, base_rate_loc=-3, sparsity=0.98
    )

    # generate datasets
    data = sample_artificial_data(theta_GT, 1000)

    nr_deleted = 0
    # remove patients with over 20 active events, so exact calculation using SSR is possible
    for nr_patient in reversed(range(data.shape[0])):
        if np.sum(data[nr_patient]) > 20:
            nr_deleted += 1
            data = np.delete(data, nr_patient, axis=0)

    print(
        f"Logging info: {nr_deleted} samples were deleted to allow for state"
        "space restriction to work"
    )

    # save ground truth theta
    if not os.path.exists(theta_folder):
        os.makedirs(theta_folder)
    np.save(os.path.join(theta_folder, f"theta_GT_{nr_dataset}.npy"), theta_GT)

    # save dataset
    if not os.path.exists(datasets_folder):
        os.makedirs(datasets_folder)
    np.save(os.path.join(datasets_folder, f"dataset_{nr_dataset}.npy"), data)
