using AdaptiveResonance
using DelimitedFiles
using Logging

# Set the log level
LogLevel(Logging.Info)

# """
#     setup_correct_partition(data::Array{T, 2}, classes::Array{T, 1}) where {T<:Real}

# Set up the data with correct partitioning for CVI tests.
# """
# function setup_correct_partition(data::Array{T, 2}, classes::Array{T, 1}) where {T<:Real}
#     # Parse the data
#     u = unique(classes)
#     n_classes = length(u)
#     dim, n_samples = get_data_shape(data)

#     # Linear normalization
#     data_norm = linear_normalization(data)
#     for ix=1:dim
#         u_values = unique(data_norm[])
#     end
# end

# Parse the data
# data_file = "data/full_data.csv"
data_file = "data/correct_partition.csv"
data = readdlm(data_file, ',')
data = permutedims(data)
train_x = data[1:2, :]
train_y = convert(Array{Int64}, data[3, :])

# Incremental
cvi = XB()
for ix = 1:length(train_y)
    param_inc!(cvi, train_x[:, ix], train_y[ix])
    evaluate!(cvi)
end

#
cvi = XB()

vind = param_batch!(cvi, train_x, train_y)
evaluate!(cvi)