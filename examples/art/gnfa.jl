using AdaptiveResonance
using Logging

# Set the log level
LogLevel(Logging.Info)
@info "GNFA Testing"

# Auxiliary generic functions for loading data, etc.
include("../../test/test_utils.jl")

# GNFA train and test
opts = opts_GNFA(rho=0.5)
my_gnfa = GNFA(opts)
# data = load_am_data(200, 50)
data = load_iris("data/Iris.csv")
local_complement_code = complement_code(data.train_x)

train!(my_gnfa, local_complement_code, y=data.train_y)
cc_test = complement_code(data.test_x)
y_hat = classify(my_gnfa, cc_test)
