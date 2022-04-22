using AdaptiveResonance
using MLDatasets

Iris.download(i_accept_the_terms_of_use=true)
features, labels = Iris.features(), Iris.labels()

art = DDVFA(rho_lb=0.6, rho_ub=0.75)

y_hat_train = train!(art, features)

art.n_categories

total_vec = [art.F2[i].n_categories for i = 1:art.n_categories]
total_cat = sum(total_vec)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

