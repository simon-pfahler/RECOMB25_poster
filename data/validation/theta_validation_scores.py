import fastmhn
import numpy as np

from parameters import *

output_file = open(theta_scores_file, "w")

output_file.write(f"Dataset mcs score\n")

# loop over datasets
for i in range(nr_datasets):
    print(f"Logging info: Now at dataset {i}")

    # load dataset
    data = np.load(f"./datasets/dataset_{i}.npy")

    # get score of exact optimization
    theta = np.load(f"./theta_learned/theta_dataset{i}_exact.npy")
    score = fastmhn.exact.gradient_and_score(theta, data)[1]

    output_file.write(f"{i} {-1} {score}\n")
    print(f"Logging info: {i} {-1} {score}")

    # get scores of approximate optimizations
    for mcs in range(2, 26):
        theta = np.load(f"./theta_learned/theta_dataset{i}_mcs{mcs}.npy")
        score = fastmhn.exact.gradient_and_score(theta, data)[1]

        output_file.write(f"{i} {mcs} {score}\n")
        print(f"Logging info: {i} {mcs} {score}")
