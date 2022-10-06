# Load the library
using AdaptiveResonance

# Create a new ART module and inspect its uninitialized data config `config`
art = FuzzyART()
art.config

fieldnames(AdaptiveResonance.DataConfig)

# Load data
using MLDatasets        # Iris dataset
using DataFrames        # DataFrames, necessary for MLDatasets.Iris()
using MLDataUtils       # Shuffling and splitting

# Get the iris dataset
iris = Iris(as_df=false)
# Manipulate the features and labels into a matrix of features and a vector of labels
features, labels = iris.features, iris.targets

labels = convertlabel(LabelEnc.Indices{Int}, vec(labels))
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

