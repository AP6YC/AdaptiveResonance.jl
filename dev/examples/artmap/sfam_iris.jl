using AdaptiveResonance # ART
using MLDatasets        # Iris dataset
using MLDataUtils       # Shuffling and splitting
using Printf            # Formatted number printing

# Get the iris dataset as a DataFrame
iris = Iris()
# Manipulate the features and labels into a matrix of features and a vector of labels
features, labels = Matrix(iris.features)', vec(Matrix{String}(iris.targets))

labels = convertlabel(LabelEnc.Indices{Int}, labels)
unique(labels)

(X_train, y_train), (X_test, y_test) = stratifiedobs((features, labels))

# Create the SFAM module
art = SFAM()

# Change the match tracking parameter after instantiation
art.opts.epsilon = 1e-2

# Train in supervised mode by directly passing the labels.
y_hat_train = train!(art, X_train, y_train)
println("Training labels: ",  size(y_hat_train), " ", typeof(y_hat_train))

# Classify both ways
y_hat = AdaptiveResonance.classify(art, X_test)
y_hat_bmu = AdaptiveResonance.classify(art, X_test, get_bmu=true)

# Check the shape and type of the output labels
println("Testing labels: ",  size(y_hat), " ", typeof(y_hat))
println("Testing labels with bmu: ",  size(y_hat_bmu), " ", typeof(y_hat_bmu))

# Calculate performance on training data, testing data, and with get_bmu
perf_train = performance(y_hat_train, y_train)
perf_test = performance(y_hat, y_test)
perf_test_bmu = performance(y_hat_bmu, y_test)

# Format each performance number for comparison
@printf "Training performance: %.4f\n" perf_train
@printf "Testing performance: %.4f\n" perf_test
@printf "Best-matching unit testing performance: %.4f\n" perf_test_bmu

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

