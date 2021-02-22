"""
    XB.jl

This is a Julia port of a MATLAB implementation of batch and incremental
Xie-Beni (XB) Cluster Validity Index.

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
using LinearAlgebra

"""
    XB

The stateful information of the Xie-Beni CVI.
"""
mutable struct XB <: AbstractCVI
    dim::Int64
    n_samples::Int64
    mu_data::Array{Float64, 1}
    n::Array{Float64, 1}
    v::Array{Float64, 2}
    CP::Array{Float64, 1}
    SEP::Float64
    G::Array{Float64, 2}
    D::Array{Float64, 2}
    WGSS::Float64
    n_clusters::Int64
    criterion_value::Array{Float64, 2}
end # XB <: AbstractCVI

"""
    XB()

Default constructor for the Xie-Beni (XB) Cluster Validity Index.
"""
function XB()
    XB(0,                               # dim
       0,                               # n_samples
       Array{Float64, 1}(undef, 0),     # mu_data
       Array{Float64, 1}(undef, 0),     # n
       Array{Float64, 2}(undef, 0, 0),  # v
       Array{Float64, 1}(undef, 0),     # CP
       0.0,                             # SEP
       Array{Float64, 2}(undef, 0, 0),  # G
       Array{Float64, 2}(undef, 0, 0),  # D
       0.0,                             # WGSS
       0,                               # n_clusters
       Array{Float64, 2}(undef, 0, 0)   # criterion_value
    )
end # XB()

function setup!(cvi::XB, sample::Array{T, 1}) where {T<:Real}
    cvi.mu_data
    # art.W_old = Array{Float64}(undef, art.config.dim_comp, 1)
    # art.W_old[:, 1] = x[:, 1]
end

"""
    param_inc!(cvi::XB, sample::Array{T, 1}, label::I) where {T<:Real, I<:Int}

Compute the XB CVI incrementally.
"""
function param_inc!(cvi::XB, sample::Array{T, 1}, label::I) where {T<:Real, I<:Int}
    n_samples_new = cvi.n_samples + 1
    if isempty(cvi.mu_data)
        mu_data_new = sample
        setup!(cvi, sample)
    else
        mu_data_new = (1 - 1/n_samples_new) .* cvi.mu_data + (1/n_samples_new) .* sample
    end

    if label > cvi.n_clusters
        n_new = 1
        v_new = sample
        CP_new = 0
        G_new = zeros(cvi.dim)
        if cvi.n_clusters == 0
            D_new = 0
        else
            D_new = zeros(cvi.n_clusters + 1, cvi.n_clusters + 1)
            D_new[1:cvi.n_clusters, 1:cvi.n_clusters] = cvi.D
            d_column_new = zeros(cvi.n_clusters + 1)
            for jx = 1:cvi.n_clusters
                d_column_new[jx] = sum((v_new - cvi.v[:, jx]).^2, dims=1)
            end
            D_new[:, label] = d_column_new
            D_new[label, :] = transpose(d_column_new)
        end
        # Update parameters
        cvi.n_clusters += 1
        cvi.n = [cvi.n; n_new]
        cvi.v = [cvi.v v_new]
        cvi.CP = [cvi.CP; CP_new]
        cvi.G = [cvi.G G_new]
        cvi.D = D_new
    else
        n_new = cvi.n[label] + 1
        v_new = (1 - 1/n_new) .* cvi.v[:, label] + (1/n_new) .* sample
        delta_v = cvi.v[:, label] - v_new
        diff_x_v = sample .- v_new
        CP_new = cvi.CP[label] + transpose(diff_x_v)*diff_x_v + cvi.n[label]*transpose(delta_v)*delta_v + 2*tranpose(delta_v)*cvi.G[:, label]
        for jx = 1:cvi.n_clusters
            if jx == label
                continue
            end
            d_column_new[jx] = sum((v_new - cvi.v[:, jx]).^2, dims=1)
        end
        # Update parameters
        cvi.n[label] = n_new
        cvi.v[:, label] = v_new
        cvi.CP[label] = CP_new
        cvi.D[:, label] = d_column_new
        cvi.D[label, :] = transpose(d_column_new)
    end
    cvi.n_samples = n_samples_new
    cvi.mu_data = mu_data_new
end # param_inc!(cvi::XB, sample::Array{T, 1}, label::I) where {T<:Real, I<:Int}

"""
    param_batch!(cvi::XB, data::Array{T, 2}, labels::Array{I, 1}) where {T<:Real, I<:Int}

Compute the XB CVI in batch.
"""
function param_batch!(cvi::XB, data::Array{T, 2}, labels::Array{I, 1}) where {T<:Real, I<:Int}
    cvi.dim, cvi.n_samples = size(data)
    cvi.mu_data = mean(data, dims=2)
    u = findfirst.(isequal.(unique(labels)), [labels])
    cvi.n_clusters = length(u)
    cvi.n = zeros(cvi.n_clusters)
    cvi.v = zeros(cvi.dim, cvi.n_clusters)
    cvi.CP = zeros(cvi.n_clusters)
    cvi.D = zeros(cvi.n_clusters, cvi.n_clusters)
    for ix = 1:cvi.n_clusters
        subset = data[:, findall(x->x==u[ix])]
        cvi.n[ix] = size(subset, 2)
        cvi.v[1:cvi.dim, ix] = mean(subset, dims=2)
        diff_x_v = subset - cvi.v[:, ix] * ones(1, cvi.n[ix])
        cvi.CP[ix] = sum(diff_x_v.^2)
    end
    for ix = 1 : (cvi.n_clusters - 1)
        for jx = ix + 1 : cvi.n_clusters
            cvi.D[jx, ix] = sum((cvi.v[:, ix] - cvi.v[:, jx]).^2, dims=1)
        end
    end
    cvi.D = cvi.D + transpose(cvi.D)
end # param_batch(cvi::XB, data::Array{Real, 2}, labels::Array{Real, 1})

function evaluate(cvi::XB)
    cvi.WGSS = sum(cvi.CP)
    if cvi.n_clusters > 1
        mask = ones(Int64, size(cvi.D))
        mask = UpperTriangular(mask)
        for ij = 1:size(mask, 1)
            mask[ij, ij] = 0
        end
        values = cvi.D[mask]
        cvi.SEP = minimum(values)
        cvi.criterion_value = cvi.WGSS/(cvi.n_samples*cvi.SEP)
    end
end # evaluate(cvi::XB)