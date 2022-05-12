# ---
# title: Unsupervised DDVFA Example
# id: ddvfa_unsupervised
# cover: ../assets/ddvfa.png
# date: 2021-11-30
# author: "[Sasha Petrenko](https://github.com/AP6YC)"
# julia: 1.6
# description: This demo shows how to use DDVFA for unsupervised learning by clustering Iris samples.
# ---

# DDVFA is an unsupervised clustering algorithm by definition, so it can be used to cluster a set of samples all at once in batch mode.

# We begin with importing AdaptiveResonance for the ART modules and MLDatasets for loading some data.
using AdaptiveResonance
using MLDatasets

# We will download the Iris dataset for its small size and benchmark use for clustering algorithms.
iris = Iris()
features, labels = Matrix(iris.features), Matrix{String}(iris.targets)

# Next, we will instantiate a DDVFA module.
# We could create an options struct for reuse with `opts=opts_DDVFA(...)`, but for now we will use the direct keyword arguments approach.
art = DDVFA(rho_lb=0.6, rho_ub=0.75)

# To train the module on the training data, we use `train!`.
# The train method returns the prescribed cluster labels, which are just what the algorithm believes are unique/separate cluster.
# This is because we are doing *unsupervised* learning rather than supervised learning with known labels.
y_hat_train = train!(art, features)

# Though we could inspect the unique entries in the list above, we can see the number of categories directly from the art module.
art.n_categories

# Because DDVFA actually has FuzzyART modules for F2 nodes, each category has its own category prototypes.
# We can see the total number of weights in the DDVFA module by summing `n_categories` across all F2 nodes.
total_vec = [art.F2[i].n_categories for i = 1:art.n_categories]
total_cat = sum(total_vec)
