"""
    SFAM.jl

Description:
    Options, structures, and logic for the Simplified Fuzzy ARTMAP (SFAM) module.

References:
    [1] G. A. Carpenter, S. Grossberg, N. Markuzon, J. H. Reynolds, and D. B. Rosen, “Fuzzy ARTMAP: A Neural Network Architecture for Incremental Supervised Learning of Analog Multidimensional Maps,” IEEE Trans. Neural Networks, vol. 3, no. 5, pp. 698-713, 1992, doi: 10.1109/72.159059.
"""

"""
    opts_SFAM()

Implements a Simple Fuzzy ARTMAP learner's options.

# Keyword Arguments
- `rho::Float`: vigilance value, [0, 1], default 0.75.
- `alpha::Float`: choice parameter, alpha > 0, default 1e-7.
- `epsilon::Float`: match tracking parameter, (0, 1), default 1e-3
- `beta::Float`: learning parameter, (0, 1], default 1.0.
- `uncommitted::Bool`: uncommitted node flag, default true.
- `display::Bool`: display flag, default true.
- `max_epoch::Int`: maximum number of epochs during training, default 1.
"""
@with_kw mutable struct opts_SFAM <: ARTOpts @deftype Float
    # Vigilance parameter: [0, 1]
    rho = 0.75; @assert rho >= 0.0 && rho <= 1.0
    # Choice parameter: alpha > 0
    alpha = 1e-7; @assert alpha > 0.0
    # Match tracking parameter
    epsilon = 1e-3; @assert epsilon > 0.0 && epsilon < 1.0
    # Learning parameter: (0, 1]
    beta = 1.0; @assert beta > 0.0 && beta <= 1.0
    # Uncommitted node flag
    uncommitted::Bool = true
    # Display flag
    display::Bool = true
    # Maximum number of epochs during training
    max_epochs::Int = 1
end # opts_SFAM()

"""
    SFAM <: ARTMAP

Simple Fuzzy ARTMAP struct.

For module options, see [`AdaptiveResonance.opts_SFAM`](@ref).

# References
1. G. A. Carpenter, S. Grossberg, N. Markuzon, J. H. Reynolds, and D. B. Rosen, “Fuzzy ARTMAP: A Neural Network Architecture for Incremental Supervised Learning of Analog Multidimensional Maps,” IEEE Trans. Neural Networks, vol. 3, no. 5, pp. 698-713, 1992, doi: 10.1109/72.159059.
"""
mutable struct SFAM <: ARTMAP
    opts::opts_SFAM
    config::DataConfig
    W::RealMatrix
    labels::IntegerVector
    n_categories::Int
    epoch::Int
end # SFAM <: ARTMAP

"""
    SFAM()

Implements a Simple Fuzzy ARTMAP learner.

# Examples
```julia-repl
julia> SFAM()
SFAM
    opts: opts_SFAM
    ...
```
"""
function SFAM()
    opts = opts_SFAM()
    SFAM(opts)
end # SFAM()

"""
    SFAM(;kwargs...)

Implements a Simple Fuzzy ARTMAP learner with keyword arguments.

# Examples
```julia-repl
julia> SFAM()
SFAM
    opts: opts_SFAM
    ...
```
"""
function SFAM(;kwargs...)
    opts = opts_SFAM(;kwargs...)
    SFAM(opts)
end # SFAM(;kwargs...)

"""
    SFAM(opts)

Implements a Simple Fuzzy ARTMAP learner with specified options.

# Examples
```julia-repl
julia> opts = opts_SFAM()
julia> SFAM(opts)
SFAM
    opts: opts_SFAM
    ...
```
"""
function SFAM(opts::opts_SFAM)
    SFAM(
        opts,                           # opts_SFAM
        DataConfig(),                   # config
        Array{Float}(undef, 0, 0),      # W
        Array{Int}(undef, 0),           # labels
        0,                              # n_categories
        0                               # epoch
    )
end # SFAM(opts::opts_SFAM)

# SFAM incremental training method
function train!(art::SFAM, x::RealVector, y::Integer ; preprocessed::Bool=false)
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

# SFAM incremental classification method
function classify(art::SFAM, x::RealVector ; preprocessed::Bool=false, get_bmu::Bool=false)
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
    stopping_conditions(art::SFAM)

Stopping conditions for Simple Fuzzy ARTMAP, checked at the end of every epoch.
"""
function stopping_conditions(art::SFAM)
    # Compute the stopping condition, return a bool
    return art.epoch >= art.opts.max_epochs
end # stopping_conditions(art::SFAM)

"""
    learn(art::SFAM, x::RealVector, W::RealVector)

Returns a single updated weight for the Simple Fuzzy ARTMAP module for weight
vector W and sample x.
"""
function learn(art::SFAM, x::RealVector, W::RealVector)
    # Update W
    return art.opts.beta .* element_min(x, W) .+ W .* (1 - art.opts.beta)
end # learn(art::SFAM, x::RealVector, W::RealVector)

"""
    activation(art::SFAM, x::RealVector, W::RealVector)

Returns the activation value of the Simple Fuzzy ARTMAP module with weight W
and sample x.
"""
function activation(art::SFAM, x::RealVector, W::RealVector)
    # Compute T and return
    return norm(element_min(x, W), 1) / (art.opts.alpha + norm(W, 1))
end # activation(art::SFAM, x::RealVector, W::RealVector)

"""
    art_match(art::SFAM, x::RealVector, W::RealVector)

Returns the match function for the Simple Fuzzy ARTMAP module with weight W and
sample x.
"""
function art_match(art::SFAM, x::RealVector, W::RealVector)
    # Compute M and return
    return norm(element_min(x, W), 1) / art.config.dim
end # art_match(art::SFAM, x::RealVector, W::RealVector)
