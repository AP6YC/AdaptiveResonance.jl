using AdaptiveResonance
using Logging

# Set the log level
LogLevel(Logging.Info)
@info "FuzzyART Testing"

# Auxiliary generic functions for loading data, etc.
include("../../test/test_utils.jl")

# FuzzyART train and test
# opts = opts_FuzzyART(rho=0.6, gamma_ref = 1.0, gamma=1.0)
opts = opts_FuzzyART(rho=0.6, gamma = 5.0)
art = FuzzyART(opts)
# data = load_am_data(200, 50)
data = load_iris("data/Iris.csv")
# local_complement_code = complement_code(data.train_x)

# train!(my_FuzzyART, local_complement_code, y=data.train_y)
train!(art, data.train_x, y=data.train_y)
# y_hat = classify(my_FuzzyART, data.test_x, get_bmu=true)
y_hat = classify(art, data.test_x)

perf = performance(y_hat, data.test_y)
@info "Performance: $perf"
