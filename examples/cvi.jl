using AdaptiveResonance
using DelimitedFiles
using Logging

# Set the log level
LogLevel(Logging.Info)

# Parse the data
data_file = "data/full_data.csv"
data = readdlm(data_file, ',')
data = permutedims(data)
train_x = data[1:2, :]
train_y = convert(Array{Int64}, data[3, :])

# Create the ART module, train, and classify
# art = SFAM()
# train!(art, train_x, train_y)

cvi = XB()

for ix = 1:length(train_y)
    vind = param_inc!(cvi, train_x[:, ix], train_y[ix])
end