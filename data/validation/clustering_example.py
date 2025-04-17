import fastmhn
import numpy as np

np.random.seed(42)

Theta = np.array(
    [
        [2, 0.3, 4, 1, 1, 1.5],
        [0.3, 1, 0.5, 1, 1, 1],
        [5, 2, 1, 1, 0.4, 1],
        [1, 1, 1, 1.5, 4, 0.2],
        [1, 1, 0.5, 3, 2, 0.1],
        [3, 1, 1, 2, 0.2, 1],
    ]
)
theta = np.log(Theta)

inv_distance_matrix = np.where(
    np.abs(theta) > np.abs(theta).T, np.abs(theta), np.abs(theta).T
)
np.fill_diagonal(inv_distance_matrix, 0)

print("Theta matrix:")
print(Theta)

print("\nDistance matrix:")
print(1 / inv_distance_matrix)

fastmhn.clustering.get_clusters(theta, 1, 1, verbose=True)
