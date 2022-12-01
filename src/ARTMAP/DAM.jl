"""
    DAM.jl

# Description:
Options, structures, and logic for the Default ARTMAP (DAM) module.

# References:
[1] G. P. Amis and G. A. Carpenter, 'Default ARTMAP 2,' IEEE Int. Conf. Neural Networks - Conf. Proc., vol. 2, no. September 2007, pp. 777-782, Mar. 2007, doi: 10.1109/IJCNN.2007.4371056.
"""

# --------------------------------------------------------------------------- #
# OPTIONS
# --------------------------------------------------------------------------- #

"""
Implements a Default ARTMAP learner's options.

$(opts_docstring)
"""
@with_kw mutable struct opts_DAM <: ARTOpts @deftype Float
    """
    Vigilance parameter: rho ∈ [0, 1].
    """
    rho = 0.75; @assert rho >= 0.0 && rho <= 1.0

    """
    Choice parameter: alpha > 0.
    """
    alpha = 1e-7; @assert alpha > 0.0

    """
    Match tracking parameter: episilon ∈ (0, 1).
    """
    epsilon = 1e-3; @assert epsilon > 0.0 && epsilon < 1.0

    """
    Learning parameter: beta ∈ (0, 1].
    """
    beta = 1.0; @assert beta > 0.0 && beta <= 1.0

    """
    Maximum number of epochs during training: ∈ [1, Inf).
    """
    max_epochs::Int = 1

    """
    Uncommitted node flag.
    """
    uncommitted::Bool = true

    """
    Display flag.
    """
    display::Bool = true
end

# --------------------------------------------------------------------------- #
# STRUCTS
# --------------------------------------------------------------------------- #

"""
Default ARTMAP struct.

For module options, see [`AdaptiveResonance.opts_DAM`](@ref).

# References
1. G. P. Amis and G. A. Carpenter, 'Default ARTMAP 2,' IEEE Int. Conf. Neural Networks - Conf. Proc., vol. 2, no. September 2007, pp. 777-782, Mar. 2007, doi: 10.1109/IJCNN.2007.4371056.
"""
mutable struct DAM <: ARTMAP
    """
    Default ARTMAP options struct.
    """
    opts::opts_DAM

    """
    Data configuration struct.
    """
    config::DataConfig

    """
    Category weight matrix.
    """
    W::Matrix{Float}

    """
    Incremental list of labels corresponding to each F2 node, self-prescribed or supervised.
    """
    labels::Vector{Int}

    """
    Number of category weights (F2 nodes).
    """
    n_categories::Int

    """
    Current training epoch.
    """
    epoch::Int
end

# --------------------------------------------------------------------------- #
# CONSTRUCTORS
# --------------------------------------------------------------------------- #

"""
Implements a Default ARTMAP learner with optional keyword arguments.

# Examples
By default:
```julia-repl
julia> DAM()
DAM
    opts: opts_DAM
    ...
```

or with keyword arguments:
```julia-repl
julia> DAM()
DAM
    opts: opts_DAM
    ...
```
"""
function DAM(;kwargs...)
    opts = opts_DAM(;kwargs...)
    DAM(opts)
end

"""
Implements a Default ARTMAP learner with specified options.

# Examples
```julia-repl
julia> opts = opts_DAM()
julia> DAM(opts)
DAM
    opts: opts_DAM
    ...
```
"""
function DAM(opts::opts_DAM)
    DAM(
        opts,                           # opts_DAM
        DataConfig(),                   # config
        Array{Float}(undef, 0, 0),      # W
        Array{Int}(undef, 0),           # labels
        0,                              # n_categories
        0                               # epoch
    )
end

# --------------------------------------------------------------------------- #
# ALGORITHMIC METHODS
# --------------------------------------------------------------------------- #

# Incremental DAM training method
function train!(art::DAM, x::RealVector, y::Integer ; preprocessed::Bool=false)
    # Run the sequential initialization procedure
    sample = init_train!(x, art, preprocessed)

    # Initialization
    if !(y in art.labels)
        # Initialize W and labels
        if isempty(art.W)
            art.W = Array{Float64}(undef, art.config.dim_comp, 1)
            art.W[:, 1] = sample
        else
            art.W = [art.W sample]
        end
        push!(art.labels, y)
        art.n_categories += 1
    else
        # Baseline vigilance parameter
        rho_baseline = art.opts.rho

        # Compute activation function
        T = zeros(art.n_categories)
        for jx in 1:art.n_categories
            T[jx] = activation(art, sample, art.W[:, jx])
        end

        # Sort activation function values in descending order
        index = sortperm(T, rev=true)
        mismatch_flag = true
        for jx in 1:art.n_categories
            # Compute match function
            M = art_match(art, sample, art.W[:, index[jx]])
            # Current winner
            if M >= rho_baseline
                if y == art.labels[index[jx]]
                    # Learn
                    @debug "Learning"
                    art.W[:, index[jx]] = learn(art, sample, art.W[:, index[jx]])
                    mismatch_flag = false
                    break
                else
                    # Match tracking
                    @debug "Match tracking"
                    rho_baseline = M + art.opts.epsilon
                end
            end
        end

        # If we triggered a mismatch
        if mismatch_flag
            # Create new weight vector
            @debug "Mismatch"
            art.W = hcat(art.W, sample)
            push!(art.labels, y)
            art.n_categories += 1
        end
    end

    # ARTMAP guarantees correct training classification, so just return the label
    return y
end

# DAM incremental classification method
function classify(art::DAM, x::RealVector ; preprocessed::Bool=false, get_bmu::Bool=false)
    # Run the sequential initialization procedure
    sample = init_classify!(x, art, preprocessed)

    # Compute activation function
    T = zeros(art.n_categories)
    for jx in 1:art.n_categories
        T[jx] = activation(art, sample, art.W[:, jx])
    end

    # Sort activation function values in descending order
    index = sortperm(T, rev=true)
    mismatch_flag = true
    for jx in 1:art.n_categories
        # Compute match function
        M = art_match(art, sample, art.W[:, index[jx]])
        # Current winner
        if M >= art.opts.rho
            y_hat = art.labels[index[jx]]
            mismatch_flag = false
            break
        end
    end

    # If we did not find a resonant category
    if mismatch_flag
        @debug "Mismatch"
        # Report either the best matching unit or the mismatch label -1
        y_hat = get_bmu ? art.labels[index[1]] : -1
    end

    return y_hat
end

"""
Stopping conditions for Default ARTMAP, checked at the end of every epoch.
"""
function stopping_conditions(art::DAM)
    # Compute the stopping condition, return a bool
    return art.epoch >= art.opts.max_epochs
end

"""
Default ARTMAP's choice-by-difference activation function.
"""
function activation(art::DAM, x::RealVector, W::RealVector)
    # Compute T and return
    return (
        norm(element_min(x, W), 1)
        + (1 - art.opts.alpha) * (art.config.dim - norm(W, 1))
    )
end

"""
Returns a single updated weight for the Default ARTMAP module for weight vector W and sample x.
"""
function learn(art::DAM, x::RealVector, W::RealVector)
    # Update W
    return art.opts.beta .* element_min(x, W) .+ W .* (1 - art.opts.beta)
end

"""
Returns the match function for the Default ARTMAP module with weight W and sample x.
"""
function art_match(art::DAM, x::RealVector, W::RealVector)
    # Compute M and return
    return norm(element_min(x, W), 1) / art.config.dim
end
