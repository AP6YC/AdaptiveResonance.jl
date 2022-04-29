# Load the library
using AdaptiveResonance

# Create a new ART module and inspect its uninitialized data config `config`
art = FuzzyART()
art.config

fieldnames(AdaptiveResonance.DataConfig)

# Load data
using MLDatasets

# We will download the Iris dataset for its small size and benchmark use for clustering algorithms.
Iris.download(i_accept_the_terms_of_use=true)
features, labels = Iris.features(), Iris.labels()

# We will then train the FuzzyART module in unsupervised mode and see that the data config is now set
y_hat_train = train!(art, features)
art.config

# Reinitialize the FuzzyART module
art = FuzzyART()
# Tell the module that we have 20 features all ranging from -1 to 1
art.config = DataConfig(-1, 1, 20)

# Assume some minimum and maximum values for each feature
mins = [-1,-2,-1.5]
maxs = [3, 2, 1]
art.config = DataConfig(mins, maxs)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

