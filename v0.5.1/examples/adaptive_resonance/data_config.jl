# Load the library
using AdaptiveResonance

# Create a new ART module and inspect its uninitialized data config `config`
art = FuzzyART()
art.config

fieldnames(AdaptiveResonance.DataConfig)

# Load data
using MLDatasets        # Iris dataset
using MLDataUtils       # Shuffling and splitting

# We will download the Iris dataset for its small size and benchmark use for clustering algorithms.
# Get the iris dataset as a DataFrame
iris = Iris()
# Manipulate the features and labels into a matrix of features and a vector of labels
features, labels = Matrix(iris.features)', vec(Matrix{String}(iris.targets))

labels = convertlabel(LabelEnc.Indices{Int}, labels)
unique(labels)

# Reinitialize the FuzzyART module
art = FuzzyART()
# Tell the module that we have 20 features all ranging from -1 to 1
art.config = DataConfig(-1, 1, 20)

# Assume some minimum and maximum values for each feature
mins = [-1,-2,-1.5]
maxs = [3, 2, 1]
art.config = DataConfig(mins, maxs)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

