# ---
# title: Write your demo in julia
# id: ddvfa_unsupervised
# cover: ../assets/ddvfa.png
# date: 2021-11-30
# author: "[Sasha Petrenko](https://github.com/AP6YC)"
# julia: 1.6
# description: This demo shows how to use DDVFA for unsupervised learning by clustering samples.
# ---

# DDVFA is an unsupervised clustering algorithm by definition, so it can be used to cluster a set of samples all at once in batch mode.

# Load some data
using MLDatasets
using AdaptiveResonance

Iris.download(i_accept_the_terms_of_use=true)
features, labels = Iris.features(), Iris.labels()

# Instantiate a DDVFA module
art = DDVFA(rho_lb=0.4, rho_ub=0.75)

# Train the module on the training data, getting the prescribed cluster labels
y_hat_train = train!(art, features)

# Total number of categories
total_vec = [art.F2[i].n_categories for i = 1:art.n_categories]
total_cat = sum(total_vec)

# Conduct inference
# y_hat_test = classify(art, test_x)