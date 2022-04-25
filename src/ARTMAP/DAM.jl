"""
    DAM.jl

Description:
    Options, structures, and logic for the Default ARTMAP (DAM) module.
"""

"""
    opts_DAM()

Implements a Default ARTMAP learner's options.

# Examples
```julia-repl
julia> my_opts = opts_DAM()
```
"""
@with_kw mutable struct opts_DAM <: ARTOpts @deftype Float
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
end # opts_DAM()

"""
    DAM <: ARTMAP

Default ARTMAP struct.
"""
mutable struct DAM <: ARTMAP
    opts::opts_DAM
    config::DataConfig
    W::RealMatrix
    labels::IntegerVector
    n_categories::Int
    epoch::Int
end # DAM <: ARTMAP

"""
    DAM()

Implements a Simple Fuzzy ARTMAP learner.

# Examples
```julia-repl
julia> DAM()
DAM
    opts: opts_DAM
    ...
```
"""
function DAM()
    opts = opts_DAM()
    DAM(opts)
end # DAM()

"""
    DAM(;kwargs...)

Implements a Default ARTMAP learner with keyword arguments.

# Examples
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
end # DAM(;kwargs...)

"""
    DAM(opts)

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
end # DAM(opts::opts_DAM)

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
    stopping_conditions(art::DAM)

Stopping conditions for Default ARTMAP, checked at the end of every epoch.
"""
function stopping_conditions(art::DAM)
    # Compute the stopping condition, return a bool
    return art.epoch >= art.opts.max_epochs
end # stopping_conditions(art::DAM)

"""
    activation(art::DAM, x::RealVector, W::RealVector)

Default ARTMAP's choice-by-difference activation function.
"""
function activation(art::DAM, x::RealVector, W::RealVector)
    # Compute T and return
    return norm(element_min(x, W), 1) +
        (1-art.opts.alpha)*(art.config.dim - norm(W, 1))
end # activation(art::DAM, x::RealVector, W::RealVector)

"""
    learn(art::DAM, x::RealVector, W::RealVector)

Returns a single updated weight for the Default ARTMAP module for weight
vector W and sample x.
"""
function learn(art::DAM, x::RealVector, W::RealVector)
    # Update W
    return art.opts.beta .* element_min(x, W) .+ W .* (1 - art.opts.beta)
end # learn(art::DAM, x::RealVector, W::RealVector)

"""
    art_match(art::DAM, x::RealVector, W::RealVector)

Returns the match function for the Default ARTMAP module with weight W and
sample x.
"""
function art_match(art::DAM, x::RealVector, W::RealVector)
    # Compute M and return
    return norm(element_min(x, W), 1) / art.config.dim
end # art_match(art::DAM, x::RealVector, W::RealVector)
