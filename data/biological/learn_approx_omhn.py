import numpy as np
import torch

torch.set_grad_enabled(False)

import fastmhn

mcs = 25

results_filename = f"theta{mcs}.dat"

# The dataset is obtained by preprocessing MSK-CHORD data:
# https://www.cbioportal.org/study/summary?id=msk_chord_2024
# It will be open-source soon.
data = np.loadtxt(
    "./G17_LUAD_large_Events.csv",
    skiprows=1,
    delimiter=",",
    dtype=int,
)
N = data.shape[0]
d = data.shape[1]

# >>> optimization parameters
nr_iterations = 50
alpha = 0.1
beta1 = 0.7
beta2 = 0.9
eps = 1e-8
reg = 3e-3
# <<< optimization parameters

theta_np = np.zeros((d + 1, d))
theta_np[:d] = fastmhn.utility.create_indep_model(data)

theta = torch.tensor(theta_np, requires_grad=True)

optimizer = torch.optim.Adam([theta], lr=alpha, betas=(beta1, beta2), eps=eps)

for t in range(nr_iterations):
    optimizer.zero_grad()

    # create MHN theta matrix equivalent to current oMHN
    ctheta = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            if i == j:
                ctheta[i, j] = theta[i, j]
            else:
                ctheta[i, j] = theta[i, j] - theta[d, j]

    g = torch.zeros(theta.shape, dtype=torch.double)
    g[:d] = -torch.from_numpy(
        fastmhn.approx.approx_gradient(
            ctheta, data, max_cluster_size=mcs, verbose=True
        )
    )

    # observation rate gradients
    g[d, :] = -torch.einsum("ij->j", g[:d] * (1 - torch.eye(g.shape[1])))

    # regularization
    for i in range(d):
        for j in range(d):
            if theta[i, j] == theta[j, i] == 0:
                continue
            g[i, j] += (
                reg
                * (2 * theta[i, j] - theta[j, i])
                / (
                    2
                    * torch.sqrt(
                        theta[i, j] ** 2
                        + theta[j, i] ** 2
                        - theta[i, j] * theta[j, i]
                    )
                )
            )
    g[d, :] += reg * torch.sign(theta[d, :])

    theta.grad = g

    print(f"{t} - {torch.linalg.norm(theta.grad)}")

    optimizer.step()
    with open(results_filename, "w") as f:
        for i in range(theta.shape[0]):
            f.write(" ".join(map(str, theta.numpy()[i])) + "\n")

with open(results_filename, "w") as f:
    for i in range(theta.shape[0]):
        f.write(" ".join(map(str, theta.numpy()[i])) + "\n")
