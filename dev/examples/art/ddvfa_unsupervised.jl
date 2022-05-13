using AdaptiveResonance # ART
using MLDatasets        # Iris dataset
using MLDataUtils       # Shuffling and splitting

# Get the iris dataset as a DataFrame
iris = Iris()
# Manipulate the features and labels into a matrix of features and a vector of labels
features, labels = Matrix(iris.features)', vec(Matrix{String}(iris.targets))

art = DDVFA(rho_lb=0.6, rho_ub=0.75)

y_hat_train = train!(art, features)

art.n_categories

total_vec = [art.F2[i].n_categories for i = 1:art.n_categories]
total_cat = sum(total_vec)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

