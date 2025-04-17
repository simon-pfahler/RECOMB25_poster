import os

import fastmhn
import numpy as np

from parameters import *

np.random.seed(42)

output_file = open(gradient_output_file, "w")

output_file.write(
    f"Dataset calc_number mcs norm(g_exact-g_approx) norm(g_exact)\n"
)

# loop over datasets
for nr_dataset in range(nr_datasets):
    # load ground truth theta and datasets
    theta_GT = np.load(os.path.join(theta_folder, f"theta_GT_{nr_dataset}.npy"))
    data = np.load(os.path.join(datasets_folder, f"dataset_{nr_dataset}.npy"))

    # loop over gradient calculations
    for nr_gradient_calculation in range(nr_gradient_calculations):
        print(
            f"Logging info: Now at dataset {nr_dataset}, "
            f"gradient {nr_gradient_calculation}, "
            f"exact calculation"
        )

        # get theta to calculate gradient at via small perturbation of theta_GT
        theta = theta_GT + np.random.normal(
            loc=0, scale=1e-1, size=theta_GT.shape
        )

        # exact gradient
        g_exact = fastmhn.exact.gradient_and_score(theta, data)[0]
        g_exact_norm = np.linalg.norm(g_exact)

        # approximations with different maximum cluster size
        for mcs in range(2, 26):
            print(
                f"Logging info: Now at dataset {nr_dataset}, "
                f"gradient {nr_gradient_calculation}, "
                f"max cluster size {mcs}"
            )
            g_approx = fastmhn.approx.approx_gradient(
                theta, data, max_cluster_size=mcs
            )

            output_file.write(
                f"{nr_dataset} {nr_gradient_calculation} {mcs} "
                f"{np.linalg.norm(g_exact-g_approx)} {g_exact_norm}\n"
            )
