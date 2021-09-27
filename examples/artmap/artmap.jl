using AdaptiveResonance

# Load some utilities, such as data loading and training/testing functions
include("../../test/test_utils.jl")

# Load the data and test across all supervised modules
data = load_iris("data/Iris.csv")

# Iterate over several ARTMAP modules
for art in [SFAM, DAM]
    # Train and classify, returning the performance
    perf = train_test_artmap(art(), data)
end
