import os

import fastmhn
import numpy as np
import torch

from parameters import *

torch.set_grad_enabled(False)

# loop over datasets
for nr_dataset in range(nr_datasets):
    print(f"Logging info: Now at dataset {nr_dataset}")

    data = np.load(os.path.join(datasets_folder, f"dataset_{nr_dataset}.npy"))
    N = data.shape[0]
    d = data.shape[1]

    results_filename = os.path.join(
        learned_theta_folder, f"theta_dataset{nr_dataset}_exact.npy"
    )

    # create initial theta from independence model
    theta = torch.tensor(
        fastmhn.utility.create_indep_model(data), requires_grad=True
    )

    optimizer = torch.optim.Adam(
        [theta], lr=alpha, betas=(beta1, beta2), eps=eps
    )

    # optimization process
    for t in range(nr_iterations):
        optimizer.zero_grad()

        # gradient of KL divergence
        g = -torch.from_numpy(
            fastmhn.exact.gradient_and_score(theta.numpy(), data)[0]
        )

        # Lasso regularization of influences
        g += reg * np.sign(theta * (1 - np.eye(theta.shape[0])))

        theta.grad = g

        print(f"{t} - {torch.linalg.norm(theta.grad)}")

        optimizer.step()

        # save intermediate result
        np.save(results_filename, theta.numpy())

    # save final result
    np.save(results_filename, theta.numpy())
