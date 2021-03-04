using AdaptiveResonance
using Logging

# Set the log level
LogLevel(Logging.Info)

# Load the examples helper functions
include("../example_functions.jl")

# Load the trainig data
train_x, train_y = get_cvi_data("data/correct_partition.csv")
n_samples = length(train_y)

# Run the CVI in incremental mode
cvi_i = XB()
for ix = 1:n_samples
    param_inc!(cvi_i, train_x[:, ix], train_y[ix])
    evaluate!(cvi_i)
end

# Run the CVI in batch mode
cvi_b = XB()
param_batch!(cvi_b, train_x, train_y)
evaluate!(cvi_b)

# Update and get the CVI at once with the porcelain functions
cvi_p = XB()
criterion_values = zeros(n_samples)
for ix = 1:n_samples
    criterion_values[ix] = get_icvi!(cvi_p, train_x[:, ix], train_y[ix])
end

# Show the last criterion value
@info "Incremental CVI value: $(cvi_i.criterion_value)"
@info "Batch CVI value: $(cvi_b.criterion_value)"
@info "Porcelain Incremental CVI value: $(criterion_values[end])"
