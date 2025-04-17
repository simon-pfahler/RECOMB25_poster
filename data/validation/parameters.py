# this parameter file is included for the generation of all validation data

# number of events in our test datasets
d = 80

# folders for ground truth thetas and datasets
theta_folder = "theta_GT"
datasets_folder = "datasets"

# output files and folders
gradient_output_file = "gradient_validation_results.dat"
learned_theta_folder = "theta_learned"
theta_scores_file = "theta_validation_score_results.dat"

# number of datasets to consider
nr_datasets = 10

# number of gradient calculations per dataset
nr_gradient_calculations = 10

# learning process parameters
nr_iterations = 50
alpha = 0.1
beta1 = 0.7
beta2 = 0.9
eps = 1e-8
reg = 1e-3
