using Revise
using AdaptiveResonance

include("lib.jl")

dim = 5
n_samples = 1000

# @profview random_test_train(2, 10)
# @profview random_test_train(dim, n_samples)
@time random_test_train(2, 10)
@time random_test_train(dim, n_samples)
