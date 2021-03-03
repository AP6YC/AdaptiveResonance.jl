"""
    PS.jl

This is a Julia port of a MATLAB implementation of batch and incremental
Xie-Beni (PS) Cluster Validity Index.

Authors:
MATLAB implementation: Leonardo Enzo Brito da Silva
Julia port: Sasha Petrenko <sap625@mst.edu>

REFERENCES
[1] X. L. Xie and G. Beni, "A Validity Measure for Fuzzy Clustering," IEEE
Transactions on Pattern Analysis and Machine Intelligence, vol. 13, no. 8,
pp. 841–847, 1991.
[2] M. Moshtaghi, J. C. Bezdek, S. M. Erfani, C. Leckie, and J. Bailey,
"Online Cluster Validity Indices for Streaming Data," ArXiv e-prints, 2018,
arXiv:1801.02937v1 [stat.ML]. [Online].
[3] M. Moshtaghi, J. C. Bezdek, S. M. Erfani, C. Leckie, J. Bailey, "Online
cluster validity indices for performance monitoring of streaming data clustering,"
Int. J. Intell. Syst., pp. 1–23, 2018.
"""

using Statistics

"""
    PS

The stateful information of the Xie-Beni CVI.
"""
mutable struct PS <: AbstractCVI
    dim::Int64
    n_samples::Int64
    mu_data::Array{Float64, 1}  # dim
    n::Array{Int64, 1}          # dim
    v::Array{Float64, 2}        # dim x n_clusters
    CP::Array{Float64, 1}       # dim
    # SEP::Float64
    S::Array{Float64, 1}        # dim
    R::Array{Float64, 2}        # dim x n_clusters
    G::Array{Float64, 2}        # dim x n_clusters
    D::Array{Float64, 2}        # n_clusters x n_clusters
    # WGSS::Float64
    n_clusters::Int64
    # criterion_value::Array{Float64, 2}
    criterion_value::Float64
end # PS <: AbstractCVI

"""
    PS()

Default constructor for the Xie-Beni (PS) Cluster Validity Index.
"""
function PS()
    PS(0,                               # dim
       0,                               # n_samples
       Array{Float64, 1}(undef, 0),     # mu_data
       Array{Int64, 1}(undef, 0),       # n
       Array{Float64, 2}(undef, 0, 0),  # v
       Array{Float64, 1}(undef, 0),     # CP
       0.0,                             # SEP
       Array{Float64, 2}(undef, 0, 0),  # G
       Array{Float64, 2}(undef, 0, 0),  # D
       0.0,                             # WGSS
       0,                               # n_clusters
       0.0                              # criterion_value
    )
end # PS()

function setup!(cvi::PS, sample::Array{T, 1}) where {T<:Real}
    # Get the feature dimension
    cvi.dim = length(sample)
    # Initialize the 2-D arrays with the correct feature dimension
    cvi.v = Array{T, 2}(undef, cvi.dim, 0)
    cvi.G = Array{T, 2}(undef, cvi.dim, 0)
end
