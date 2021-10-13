"""
    dvfa_supervised.jl

Description:
    Train and test DVFA in a supervised fashion on the Iris dataset.

Though DVFA is by definition an unsupervised clustering algorithm, a simple supervised learning mode can be done by replacing the internal incremental cluster labels with supervisory ones.
This is done to get a general understanding of the algorithm's performance in cases where supervised labels are available.
"""

using Logging
using DelimitedFiles
using AdaptiveResonance

# Set the log level
LogLevel(Logging.Info)

# Add the local utility functions
include("../../test/test_utils.jl")

# Load the data and test across all supervised modules
data = load_iris("data/Iris.csv")

# Instantiate the DVFA module with default options
art = DVFA()

# Train with the simple supervised mode
y_hat_train = train!(art, data.train_x, y=data.train_y)

# Classify with and without the "best matching unit" option to compare performances
y_hat = classify(art, data.test_x)
y_hat_bmu = classify(art, data.test_x, get_bmu=true)

# Calculate performance
perf_train = performance(y_hat_train, data.train_y)
perf_test = performance(y_hat, data.test_y)
perf_test_bmu = performance(y_hat_bmu, data.test_y)

# Log the supervised training performances
@info "DVFA Training Perf: $perf_train"
@info "DVFA Testing Perf: $perf_test"
@info "DVFA Testing BMU Perf: $perf_test_bmu"
