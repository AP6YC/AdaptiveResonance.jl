# ---
# title: ART Options Example
# id: options
# cover: options-cover.png
# date: 2021-12-2
# author: "[Sasha Petrenko](https://github.com/AP6YC)"
# julia: 1.6
# description: This demo illustrates how to use options and modify the options for all ART and ARTMAP modules.
# ---

# ## Overview

# The `AdaptiveResonance.jl` package has several ways of handling options for ART modules.
# These methods are meant to give maximum flexibility to the user for sharing and interpreting options, which themselves vary between each module.

# !!! note
#     For more info on options in ART modules, see the guide in the docs on [ART options](@ref art_options).

# ## ART Options

# To get a feel for the ART options system, we will inspect different options and their instantiation methods.

# ### Inspection

# First, we load `AdaptiveResonance`:
using AdaptiveResonance

# Every ART module has a default constructor, which can be instantiated in the usual way:

## Create a FuzzyART module with default options
my_fuzzyart = FuzzyART()
typeof(my_fuzzyart)

# Within every ART module is a [Parameters.jl](https://github.com/mauro3/Parameters.jl) struct named `opts` containing the options for the module

## Check the FuzzyART options
my_fuzzyart.opts

# Note that the options here have the type `opts_FuzzyART`.
# This nomenclature is used throughout the module to indicate an options type associated with an ART module.
# For example, the options for a DDVFA module are `opts_DDVFA`:

## Create a DDVFA module and check the type of the options
my_ddvfa = DDVFA()
typeof(my_ddvfa.opts)

# In fact, we can create an instance of these options with a default constructor:

## Create a separate options struct
my_fuzzyart_opts = opts_FuzzyART()

# In addition to the default constructor, we can construct ART modules by instantiating these options and passing them to the module during construction:

## Instantiate an ART module by passing our options
my_fuzzyart = FuzzyART(my_fuzzyart_opts)
my_other_fuzzyart = FuzzyART(my_fuzzyart_opts)

# ### Specifying Options

# Now to the good stuff: because of the behavior of the `Parameters.jl` type, each option has a default value that we can modify during instantiation with keyword arguments:

## Change some of the default FuzzyART options
my_fuzzyart_opts = opts_FuzzyART(
    rho=0.6,
    gamma_normalization=true
)
my_fuzzyart = FuzzyART(my_fuzzyart_opts)

# As some syntactic sugar, we can pass these keyword arguments directly to the module during instantiation if we have no need to share option structs:

## Pass these keyword arguments to the module directly
my_fuzzyart = FuzzyART(
    rho=0.6,
    gamma_normalization=true
)

# Before training, we can also instantiate the model and alter the options afterward:
my_fuzzyart = FuzzyART()
my_fuzzyart.opts.rho=0.6

# !!! note
#     All ART modules are designed to use this options struct internally when the parameters are needed.
#     It is possible to change these parameters in the middle of training and evaluation, but some algorithmic instability may occur.

# ## Comparison

# To see the effect that changing these parameters has on the modules, we can train and test them side-by-side.

# We begin with importing AdaptiveResonance for the ART modules and MLDatasets for some data utilities.
using MLDatasets        # Iris dataset
using MLDataUtils       # Shuffling and splitting
using Printf            # Formatted number printing
using MultivariateStats # Principal component analysis (PCA)
using Plots             # Plotting frontend
pyplot()                # Use PyPlot backend

# We will download the Iris dataset for its small size and benchmark use for clustering algorithms.
Iris.download(i_accept_the_terms_of_use=true)
features, labels = Iris.features(), Iris.labels()

# Because the MLDatasets package gives us Iris labels as strings, we will use the `MLDataUtils.convertlabel` method with the `MLLabelUtils.LabelEnc.Indices` type to get a list of integers representing each class:
labels = convertlabel(LabelEnc.Indices{Int}, labels)
unique(labels)

# Next, we will create a train/test split with the `MLDataUtils.stratifiedobs` utility:
(X_train, y_train), (X_test, y_test) = stratifiedobs((features, labels))

# Now we can create several FuzzyART modules with different options.

## Create two FuzzyARTs with different vigilance values and suppressing logging messages
rho_1 = 0.5
rho_2 = 0.7
my_fuzzyart_1 = FuzzyART(rho=rho_1, display=false)
my_fuzzyart_2 = FuzzyART(rho=rho_2, display=false)

# Here, we will train these FuzzyART modules in simple supervised mode by passing the supervised labels as a keyword argument:

## Train in simple supervised mode by passing the labels as a keyword argument.
y_hat_train_1 = train!(my_fuzzyart_1, X_train, y=y_train)
y_hat_train_2 = train!(my_fuzzyart_2, X_train, y=y_train)

# We then classify the test data with both modules:
y_hat_1 = AdaptiveResonance.classify(my_fuzzyart_1, X_test, get_bmu=true)
y_hat_2 = AdaptiveResonance.classify(my_fuzzyart_2, X_test, get_bmu=true)

## Check the shape and type of the output labels
println("FuzzyART 1 labels: ",  size(y_hat_1), " ", typeof(y_hat_1))
println("FuzzyART 2 labels: ",  size(y_hat_2), " ", typeof(y_hat_2))

## Calculate the performance on the test data
perf_test_1 = performance(y_hat_1, y_test)
perf_test_2 = performance(y_hat_2, y_test)

## Format each performance number for comparison
@printf "Testing performance rho=%.1f: %.4f\n" rho_1 perf_test_1
@printf "Testing performance rho=%.1f: %.4f\n" rho_2 perf_test_2

# In addition to having different performances, we can see that there is a subsequent trade-off in the number of categories used:
## Print the number of categories for each vigilance parameter
@printf "Number of categories rho=%.1f: %i\n" rho_1 my_fuzzyart_1.n_categories
@printf "Number of categories rho=%.1f: %i\n" rho_2 my_fuzzyart_2.n_categories

# The variation between vigilance parameter, number of categories created during learning, and testing performance/generalization is a central theme in ART-based algorithms.

# ## Visualization

# Now, to visualize how the two models differ in how they partition the data, we can use principal component analysis (PCA) to compress to two plotting dimensions.
# PCA is a method to represent a dataset in a different number of dimensions while preserving the relative separation between datapoints.
# Though most datasets are not able to be effectively transformed down to two dimensions, this technique is useful to get a general sense of how well separated the classes are and how well your algorithm classifies them.
## Train a PCA model to visually separate the features in two dimensions.
M = fit(PCA, features; maxoutdim=2)

## Apply the PCA model to the testing set
X_test_pca = transform(M, X_test)

# We can now plot the PCA'ed test set and label them according to the two FuzzyART's

## Create the two scatterplot objects
p1 = scatter(
    X_test_pca[1, :],
    X_test_pca[2, :],
    group=y_hat_1,
    markersize=8,
    title=@sprintf "FuzzyART \$\\rho\$ = %.1f" rho_1
)
p2 = scatter(
    X_test_pca[1, :],   # PCA dimension 1
    X_test_pca[2, :],   # PCA dimension 2
    group = y_hat_2,    # labels belonging to each point
    markersize = 8,     # size of scatter points
    title=@sprintf "FuzzyART \$\\rho\$ = %.1f" rho_2    # formatted title
)

## Plot the two scatterplots together
plot(
    p1, p2,                 # scatterplot objects
    layout = (1, 2),        # plot side-by-side
    legend = false,         # no legend
    xtickfontsize = 12,     # x-tick size
    ytickfontsize = 12,     # y-tick size
    dpi = 300,              # Set the dots-per-inch
    xlims = :round,         # Round up the x-limits to the nearest whole number
    xlabel = "\$PCA_1\$",   # x-label
    ylabel = "\$PCA_2\$",   # y-label
)

# We can see that the two different vigilance values result in similar resutls on the whole, though they differ in how they classify certain samples that straddle the border between
png("options-cover") #hide
