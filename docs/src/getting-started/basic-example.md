# Basic Example

This page demonstrates a full basic example of an `AdaptiveResonance.jl` workflow.
In the example below, we create a dataset generated from two multivariate Gaussian distributions in two dimensions, showing how an ART module can be used in unsupervised or simple supervised modes alongside an ARTMAP module that is explicitly supervised-only.

For more examples that you can run yourself in Julia notebooks, see the [Examples](@ref examples) page.

```@example
# Copyright © 2021 Alexander L. Hayes
# MIT License

using AdaptiveResonance
using Distributions, Random
using MLUtils
using Plots

"""
Demonstrates Unsupervised DDVFA, Supervised DDVFA, and (Supervised) SFAM on a toy problem
with two multivariate Gaussians.
"""

# Setup two multivariate Gaussians and sampling 1000 points from each.

rng = MersenneTwister(1234)
dist1 = MvNormal([0.0, 6.0], [1.0 0.0; 0.0 1.0])
dist2 = MvNormal([4.5, 6.0], [2.0 -1.5; -1.5 2.0])

N_POINTS = 1000

X = hcat(rand(rng, dist1, N_POINTS), rand(rng, dist2, N_POINTS))
y = vcat(ones(Int64, N_POINTS), zeros(Int64, N_POINTS))

# Show the original data in its own plot
p1 = scatter(X[1,:], X[2,:], group=y, title="Original Data")

(X_train, y_train), (X_test, y_test) = splitobs((X, y); at=0.7, shuffle=true, stratified=y)

# Standardize data types
X_train = convert(Matrix{Float64}, X_train)
X_test = convert(Matrix{Float64}, X_test)
y_train = convert(Vector{Int}, y_train)
y_test = convert(Vector{Int}, y_test)

# Get the ranges of each feature dimension for a consistent DataConfig between modules
mins = minimum(X, dims=2) |> vec
maxs = maximum(X, dims=2) |> vec
config = DataConfig(mins, maxs)

# Unsupervised DDVFA
art = DDVFA()
art.config = config
train!(art, X_train)
y_hat_test = AdaptiveResonance.classify(art, X_test, get_bmu=true)
p2 = scatter(X_test[1,:], X_test[2,:], group=y_hat_test, title="Unsupervised DDVFA")

# Supervised DDVFA
art = DDVFA()
art.config = config
train!(art, X_train, y=y_train)
y_hat_test = AdaptiveResonance.classify(art, X_test, get_bmu=true)
p3 = scatter(X_test[1,:], X_test[2,:], group=y_hat_test, title="Supervised DDVFA",
    xlabel="Performance: " * string(round(performance(y_hat_test, y_test); digits=3)))

# Supervised SFAM
art = SFAM()
art.config = config
train!(art, X_train, y_train)
y_hat_test = AdaptiveResonance.classify(art, X_test, get_bmu=true)
p4 = scatter(X_test[1,:], X_test[2,:], group=y_hat_test, title="Supervised SFAM",
    xlabel="Performance: " * string(round(performance(y_hat_test, y_test); digits=3)))

# Performance Measure + display the plots
plot(p1, p2, p3, p4, layout=(1, 4), legend = false, xtickfontsize=6, xguidefontsize=8, titlefont=font(8))
```
