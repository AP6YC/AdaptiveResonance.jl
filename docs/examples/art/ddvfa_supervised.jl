# ---
# title: Supervised DDVFA Example
# id: ddvfa_supervised
# cover: ../assets/ddvfa.png
# date: 2021-11-30
# author: "[Sasha Petrenko](https://github.com/AP6YC)"
# julia: 1.6
# description: This demo shows how to use DDVFA for simple supervised learning by clustering Iris samples and mapping the modules internal categories to the true labels.
# ---

# DDVFA is an unsupervised clustering algorithm by definition, but it can be adaptived for supervised learning by mapping the module's internal categories to the true labels.
# ART modules such as DDVFA can also be used in simple supervised mode where provided labels are used in place of internal incremental labels for the clusters, providing a method of assessing the clustering performance when labels are available.

# We begin with importing AdaptiveResonance for the ART modules and MLDatasets for some data utilities.
using AdaptiveResonance # ART
using MLDatasets        # Iris dataset
using MLDataUtils       # Shuffling and splitting
using Printf            # Formatted number printing

# We will download the Iris dataset for its small size and benchmark use for clustering algorithms.
## Get the iris dataset as a DataFrame
iris = Iris()
## Manipulate the features and labels into a matrix of features and a vector of labels
features, labels = Matrix(iris.features)', vec(Matrix{String}(iris.targets))

# Because the MLDatasets package gives us Iris labels as strings, we will use the `MLDataUtils.convertlabel` method with the `MLLabelUtils.LabelEnc.Indices` type to get a list of integers representing each class:
labels = convertlabel(LabelEnc.Indices{Int}, labels)
unique(labels)

# Next, we will create a train/test split with the `MLDataUtils.stratifiedobs` utility:
(X_train, y_train), (X_test, y_test) = stratifiedobs((features, labels))

# Now, we can create our DDVFA module.
# We'll do so with the default contstructor, though the module itself has many options that you can alter during instantiation.
art = DDVFA()

# We can train the model in batch mode upon the data in a simple supervised mode.
# We do so by passing the integer vector of labels to the training method with the simple keyword `y`.
# Just as in unsupervised training, we can extract the module's prescribed labels from the training method, which should match up to the training labels as we will see later.

## Train in simple supervised mode by passing the labels as a keyword argument.
y_hat_train = train!(art, X_train, y=y_train)
println("Training labels: ",  size(y_hat_train), " ", typeof(y_hat_train))

# We can classify the testing data to see how we generalize.
# At the same time, we can see the effect of getting the best-matching unit in the case of complete mismatch (see the docs on [Mismatch vs. BMU](@ref mismatch-bmu))

## Classify both ways
y_hat = AdaptiveResonance.classify(art, X_test)
y_hat_bmu = AdaptiveResonance.classify(art, X_test, get_bmu=true)

## Check the shape and type of the output labels
println("Testing labels: ",  size(y_hat), " ", typeof(y_hat))
println("Testing labels with bmu: ",  size(y_hat_bmu), " ", typeof(y_hat_bmu))

# Finally, we can calculate the performances (number correct over total) of the model upon all three regimes:
# 1. Training data
# 2. Testing data
# 2. Testing data with `get_bmu=true`

## Calculate performance on training data, testing data, and with get_bmu
perf_train = performance(y_hat_train, y_train)
perf_test = performance(y_hat, y_test)
perf_test_bmu = performance(y_hat_bmu, y_test)

## Format each performance number for comparison
@printf "Training performance: %.4f\n" perf_train
@printf "Testing performance: %.4f\n" perf_test
@printf "Best-matching unit testing performance: %.4f\n" perf_test_bmu
