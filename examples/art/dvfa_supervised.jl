"""
    ddvfa_supervised.jl

Description:
    Train and test DDVFA in a supervised fashion on the Iris dataset.
"""

using Logging
using DelimitedFiles
using AdaptiveResonance
using Random

Random.seed!(0)

# Set the log level
LogLevel(Logging.Info)

# Add the local utility functions
include("../../test/test_utils.jl")

# Load the data and test across all supervised modules
data = load_iris("data/Iris.csv")

# Train and classify
art = DVFA()
y_hat_train = train!(art, data.train_x, y=data.train_y)
y_hat = classify(art, data.test_x)
y_hat_bmu = classify(art, data.test_x, get_bmu=true)

# Calculate performance
perf_train = performance(y_hat_train, data.train_y)
perf_test = performance(y_hat, data.test_y)
perf_test_bmu = performance(y_hat_bmu, data.test_y)

@info "DVFA Training Perf: $perf_train"
@info "DVFA Testing Perf: $perf_test"
@info "DVFA Testing BMU Perf: $perf_test_bmu"
