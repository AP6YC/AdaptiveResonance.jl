using Logging
using DelimitedFiles
using AdaptiveResonance

# using Profile
# using ProfileVega

# Set the log level
LogLevel(Logging.Info)

include("../../test/test_utils.jl")

# Load the data and test across all supervised modules
data = load_iris("data/Iris.csv")

# Train and classify
art = DDVFA()
y_hat_train = train!(art, data.train_x, y=data.train_y)
y_hat = classify(art, data.test_x)

# Calculate performance
perf_train = performance(y_hat_train, data.train_y)
perf_test = performance(y_hat, data.test_y)
# perf_training = performance(y_hat_train_old, y_hat_train)

@info "DDVFA Training Perf: $perf_train"
@info "DDVFA Testing Perf: $perf_test"
# @info "DDVFA Training Discrepancy: $perf_training"