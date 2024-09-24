using AdaptiveResonance # ART
using MLDatasets        # Iris dataset
using DataFrames        # DataFrames, necessary for MLDatasets.Iris()
using MLDataUtils       # Shuffling and splitting
using Printf            # Formatted number printing

# Get the iris dataset
iris = Iris(as_df=false)
# Manipulate the features and labels into a matrix of features and a vector of labels
features, labels = iris.features, iris.targets

labels = convertlabel(LabelEnc.Indices{Int}, vec(labels))
unique(labels)

(X_train, y_train), (X_test, y_test) = stratifiedobs((features, labels))

# Create several modules for batch and incremental training.
# We can take advantage of the options instantiation method here to use the same options for both modules.
opts = opts_DDVFA(rho_lb=0.6, rho_ub=0.75)
art_batch = DDVFA(opts)
art_incremental = DDVFA(opts)

# Setup the data config on all of the features.
data_setup!(art_incremental.config, features)

y_hat_batch_train = train!(art_batch, X_train, y=y_train)
println("Training labels: ",  size(y_hat_batch_train), " ", typeof(y_hat_batch_train))

# Get the number of training samples
n_train = length(y_train)
# Create a container for the training output labels
y_hat_incremental_train = zeros(Int, n_train)
# Iterate over all training samples
for ix in eachindex(y_train)
    sample = X_train[:, ix]
    label = y_train[ix]
    y_hat_incremental_train[ix] = train!(art_incremental, sample, y=label)
end

# Classify one model in batch mode
y_hat_batch = AdaptiveResonance.classify(art_batch, X_test, get_bmu=true)

# Classify one model incrementally
n_test = length(y_test)
y_hat_incremental = zeros(Int, n_test)
for ix = 1:n_test
    y_hat_incremental[ix] = AdaptiveResonance.classify(art_incremental, X_test[:, ix], get_bmu=true)
end

# Check the shape and type of the output labels
println("Batch testing labels: ",  size(y_hat_batch), " ", typeof(y_hat_batch))
println("Incremental testing labels: ",  size(y_hat_incremental), " ", typeof(y_hat_incremental))

# Calculate performance on training data, testing data, and with get_bmu
perf_train_batch = performance(y_hat_batch_train, y_train)
perf_train_incremental = performance(y_hat_incremental_train, y_train)
perf_test_batch = performance(y_hat_batch, y_test)
perf_test_incremental = performance(y_hat_incremental, y_test)

# Format each performance number for comparison
@printf "Batch training performance: %.4f\n" perf_train_batch
@printf "Incremental training performance: %.4f\n" perf_train_incremental
@printf "Batch testing performance: %.4f\n" perf_test_batch
@printf "Incremental testing performance: %.4f\n" perf_test_incremental

# Import visualization utilities
using Printf            # Formatted number printing
using MultivariateStats # Principal component analysis (PCA)
using Plots             # Plotting frontend
gr()                    # Use the default GR backend explicitly

# Train a PCA model
M = fit(PCA, features; maxoutdim=2)

# Apply the PCA model to the testing set
X_test_pca = MultivariateStats.transform(M, X_test)

# Create a scatterplot object from the data with some additional formatting options
scatter(
    X_test_pca[1, :],       # PCA dimension 1
    X_test_pca[2, :],       # PCA dimension 2
    group = y_hat_batch,    # labels belonging to each point
    markersize = 8,         # size of scatter points
    legend = false,         # no legend
    xtickfontsize = 12,     # x-tick size
    ytickfontsize = 12,     # y-tick size
    dpi = 300,              # Set the dots-per-inch
    xlims = :round,         # Round up the x-limits to the nearest whole number
    xlabel = "\$PCA_1\$",   # x-label
    ylabel = "\$PCA_2\$",   # y-label
    title = (@sprintf "DDVFA Iris Clusters"),   # formatted title
)

png("assets/incremental-batch-cover") #hide

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
