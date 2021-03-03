using Revise
using AdaptiveResonance
using DelimitedFiles
using Logging

# Set the log level
LogLevel(Logging.Info)

# Parse the data
data_file = "data/correct_partition.csv"
data = readdlm(data_file, ',')
data = permutedims(data)
train_x = data[1:2, :]
train_y = convert(Array{Int64}, data[3, :])

# Incremental
cvi_i = CONN()
art = SFAM()
train!(art, train_x, train_y)
for ix = 1:length(train_y)
    param_inc!(cvi_i, train_x[:, ix], train_y[ix])
    evaluate!(cvi_i)
end

# Batch
# cvi_b = CONN()

# vind = param_batch!(cvi_b, train_x, train_y)
# evaluate!(cvi_b)