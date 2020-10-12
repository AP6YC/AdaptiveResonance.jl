module ARTMAP

using Parameters
using LinearAlgebra
using Logging
using ProgressBars
using Printf

using LinearAlgebra
using MLJ

# export SFAM
# include("funcs.jl")

@with_kw mutable struct opts_SFAM @deftype Float64
    # Vigilance parameter: [0, 1]
    rho = 0.75; @assert rho >= 0 && rho <= 1
    # Choice parameter: alpha > 0
    alpha = 1e-7; @assert alpha > 0
    # Match tracking parameter
    epsilon = 1e-3; @assert epsilon > 0 && epsilon < 1
    # Learning parameter: (0, 1]
    beta = 1; @assert beta > 0 && beta <= 1
    # Similarity method (activation and match):
    #   'single', 'average', 'complete', 'median', 'weighted', or 'centroid'
    method::String = "single"
    # Display flag
    display::Bool = true
    # shuffle::Bool = false
    random_seed = 1234.5678
    max_epochs = 1
end # opts_SFAM

mutable struct SFAM
    opts::opts_SFAM
    W::Array{Float64, 2}
    W_old::Array{Float64, 2}
    labels::Array{Int, 1}
    y::Array{Int, 1}
    dim::Int
    n_categories::Int
    epoch::Int
end

function SFAM()
    opts = opts_SFAM()
    SFAM(opts,
         Array{Float64}(undef, 0,0),
         Array{Float64}(undef, 0,0),
         Array{Int}(undef, 0),
         Array{Int}(undef, 0),
         0, 0, 0)
end

function SFAM(opts::opts_SFAM)
    SFAM(opts, [], [], [], [], 0, 0, 0)
end

function train!(art::SFAM, x::Array, y::Array)
    art.dim, n_samples = size(x)
    art.y = zeros(Int, n_samples)
    x = complement_code(x)
    art.epoch = 0

    while true
        art.epoch += 1
        iter = ProgressBar(1:n_samples)
        for ix in iter
            set_description(iter, string(@sprintf("Ep: %i, ID: %i, Cat: %i", art.epoch, ix, art.n_categories)))
            if !(y[ix] in art.labels)
                # Initialize W and labels
                if isempty(art.W)
                    art.W = Array{Float64}(undef, art.dim*2, 1)
                    art.W_old = Array{Float64}(undef, art.dim*2, 1)
                    art.W[:, ix] = x[:, ix]
                else
                    art.W = [art.W x[:, ix]]
                end
                push!(art.labels, y[ix])
                art.n_categories += 1
                art.y[ix] = y[ix]
            else
                # Baseline vigilance parameter
                rho_baseline = art.opts.rho

                # Compute activation function
                T = zeros(art.n_categories)
                for jx in 1:art.n_categories
                    T[jx] = activation(art, x[:, ix], art.W[:, jx])
                end

                # Sort activation function values in descending order
                index = sortperm(T, rev=true)
                mismatch_flag = true
                for jx in 1:art.n_categories
                    # Compute match function
                    M = art_match(art, x[:, ix], art.W[:, index[jx]])
                    @debug M
                    # Current winner
                    if M >= rho_baseline
                        if y[ix] == art.labels[index[jx]]
                            # Learn
                            @debug "Learning"
                            art.W[:, index[jx]] = learn(art, x[:, ix], art.W[:, index[jx]])
                            art.y[ix] = art.labels[index[jx]]
                            mismatch_flag = false
                            break
                        else
                            # Match tracking
                            @debug "Match tracking"
                            rho_baseline = M + art.opts.epsilon
                        end
                    end
                end
                if mismatch_flag
                    # Create new weight vector
                    @debug "Mismatch"
                    art.W = hcat(art.W, x[:, ix])
                    push!(art.labels, y[ix])
                    art.n_categories += 1
                    art.y[ix] = y[ix]
                end
            end
        end
        if stopping_conditions(art)
            break
        end
        art.W_old = deepcopy(art.W)
    end
end

function classify(art::SFAM, x::Array)
    art.dim, n_samples = size(x)
    y_hat = zeros(Int, n_samples)
    x = complement_code(x)

    iter = ProgressBar(1:n_samples)
    for ix in iter
        set_description(iter, string(@sprintf("ID: %i, Cat: %i", ix, art.n_categories)))

        # Compute activation function
        T = zeros(art.n_categories)
        for jx in 1:art.n_categories
            T[jx] = activation(art, x[:, ix], art.W[:, jx])
        end

        # Sort activation function values in descending order
        index = sortperm(T, rev=true)
        mismatch_flag = true
        for jx in 1:art.n_categories
            # Compute match function
            M = art_match(art, x[:, ix], art.W[:, index[jx]])
            @debug M
            # Current winner
            if M >= art.opts.rho
                y_hat[ix] = art.labels[index[jx]]
                mismatch_flag = false
                break
            end
        end
        if mismatch_flag
            # Create new weight vector
            @debug "Mismatch"
            y_hat[ix] = -1
        end
    end
    return y_hat
end

function complement_code(data::Array)
    # Complement code the data and return a concatenated matrix
    dim, n_samples = size(data)
    x_raw = zeros(dim, n_samples)

    mins = [minimum(data[i, :]) for i in 1:dim]
    maxs = [maximum(data[i, :]) for i in 1:dim]

    for i = 1:dim
        if maxs[i] - mins[i] != 0
            x_raw[i, :] = (data[i, :] .- mins[i]) ./ (maxs[i] - mins[i])
        end
    end

    x = vcat(x_raw, 1 .- x_raw)
    return x
end

function stopping_conditions(art::SFAM)
    # Compute the stopping condition, return a bool
    # stop = false
    # if art.W == art.W_old || art.epoch >= art.opts.max_epochs
    #     stop = true
    # end
    # return stop
    return art.W == art.W_old || art.epoch >= art.opts.max_epochs
end

function element_min(x::Array, W::Array)
    # Compute the element-wise minimum of two vectors
    return minimum([x W], dims = 2)
end

function learn(art::SFAM, x::Array, W::Array)
    # Update W
    return art.opts.beta .* element_min(x, W) .+ W .* (1 - art.opts.beta)
end

function activation(art::SFAM, x::Array, W::Array)
    # Compute T and return
    return norm(element_min(x, W), 1) / (art.opts.alpha + norm(W, 1))
end

function art_match(art::SFAM, x::Array, W::Array)
    # Compute M and return
    return norm(element_min(x, W), 1) / art.dim
end

function performance(y_hat::Array, y::Array)
    # Compute the confusion matrix and calculate performance as trace/sum
    conf = confusion_matrix(categorical(y_hat), categorical(y), warn=false)
    perf = tr(conf.mat)/sum(conf.mat)
end

# using Parameters

# export SFAM

# include("funcs.jl")

# @with_kw mutable struct opts_SFAM @deftype Float64
#     # Vigilance parameter: [0, 1]
#     rho = 0.6; @assert rho >= 0 && rho <= 1
#     # Choice parameter: alpha > 0
#     alpha = 1e-3; @assert alpha > 0
#     # Learning parameter: (0, 1]
#     beta = 1; @assert beta > 0 && beta <= 1
#     # Similarity method (activation and match):
#     #   'single', 'average', 'complete', 'median', 'weighted', or 'centroid'
#     method::String = "single"
#     # Display flag
#     display::Bool = true
#     # shuffle::Bool = false
#     random_seed = 1234.5678
# end # opts_SFAM

# mutable struct SFAM
#     opts
#     W
#     W_old
#     labels
#     dim
#     n_categories
# end

# function SFAM()
#     opts = opts_SFAM()
#     SFAM(opts, [], [], [], [], [])
# end

# function SFAM(opts)
#     SFAM(opts, [], [], [], [], [])
# end

end