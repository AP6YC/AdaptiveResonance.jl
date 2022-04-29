"""
    DVFA.jl

Description:
    Includes all of the structures and logic for running a Dual-Vigilance Fuzzy ART (DVFA) module.

Authors:
    MATLAB implementation: Leonardo Enzo Brito da Silva
    Julia port: Sasha Petrenko <sap625@mst.edu>

References:
[1] L. E. Brito da Silva, I. Elnabarawy and D. C. Wunsch II, "Dual
Vigilance Fuzzy ART," Neural Networks Letters. To appear.
[2] G. Carpenter, S. Grossberg, and D. Rosen, "Fuzzy ART: Fast
stable learning and categorization of analog patterns by an adaptive
resonance system," Neural Networks, vol. 4, no. 6, pp. 759-771, 1991.
"""

"""
    opts_DVFA()

Dual Vigilance Fuzzy ART options struct.

# Keyword Arguments
- `rho_lb::Float`: lower-bound vigilance value, [0, 1], default 0.55.
- `rho_ub::Float`: upper-bound vigilance value, [0, 1], default 0.75.
- `alpha::Float`: choice parameter, alpha > 0, default 1e-3.
- `beta::Float`: learning parameter, (0, 1], default 1.0.
- `display::Bool`: display flag, default true.
- `max_epoch::Int`: maximum number of epochs during training, default 1.
"""
@with_kw mutable struct opts_DVFA <: ARTOpts @deftype Float
    # Lower-bound vigilance parameter: [0, 1]
    rho_lb = 0.55; @assert rho_lb >= 0.0 && rho_lb <= 1.0
    # Upper bound vigilance parameter: [0, 1]
    rho_ub = 0.75; @assert rho_ub >= 0.0 && rho_ub <= 1.0
    # Choice parameter: alpha > 0
    alpha = 1e-3; @assert alpha > 0.0
    # Learning parameter: (0, 1]
    beta = 1.0; @assert beta > 0.0 && beta <= 1.0
    # Display flag
    display::Bool = true
    # Maximum number of epochs during training
    max_epochs::Int = 1
end # opts_DVFA

"""
    DVFA <: ART

Dual Vigilance Fuzzy ARTMAP module struct.

For module options, see [`AdaptiveResonance.opts_DVFA`](@ref).

# Option Parameters
- `opts::opts_DVFA`: DVFA options struct.
- `config::DataConfig`: data configuration struct.

# Working Parameters
- `threshold_ub::Float`: operating upper bound module threshold value, a function of the upper bound vigilance parameter.
- `threshold_lb::Float`: operating lower bound module threshold value, a function of the lower bound vigilance parameter.
- `labels::IntegerVector`: incremental list of labels corresponding to each F2 node, self-prescribed or supervised.
- `W::RealMatrix`: category weight matrix.
- `T::RealVector`: activation values for every weight for a given sample.
- `M::RealVector`: match values for every weight for a given sample.
- `n_categories::Int`: number of category weights (F2 nodes).
- `n_clusters::Int`: number of labeled clusters, may be lower than `n_categories`
- `epoch::Int`: current training epoch.

# References:
1. L. E. Brito da Silva, I. Elnabarawy and D. C. Wunsch II, "Dual Vigilance Fuzzy ART," Neural Networks Letters. To appear.
2. G. Carpenter, S. Grossberg, and D. Rosen, "Fuzzy ART: Fast stable learning and categorization of analog patterns by an adaptive resonance system," Neural Networks, vol. 4, no. 6, pp. 759-771, 1991.
"""
mutable struct DVFA <: ART
    # Get parameters
    opts::opts_DVFA
    config::DataConfig

    # Working variables
    threshold_ub::Float
    threshold_lb::Float
    labels::IntegerVector
    W::RealMatrix
    T::RealVector
    M::RealVector
    n_categories::Int
    n_clusters::Int
    epoch::Int
end # DVFA

"""
    DVFA()

Implements a DVFA learner with default options.

# Examples
```julia-repl
julia> DVFA()
DVFA
    opts: opts_DDVFA
    ...
```
"""
function DVFA()
    opts = opts_DVFA()
    DVFA(opts)
end # DVFA()

"""
    DVFA(;kwargs...)

Implements a DVFA learner with keyword arguments.

# Examples
```julia-repl
julia> DVFA(rho=0.7)
DVFA
    opts: opts_DDVFA
    ...
```
"""
function DVFA(;kwargs...)
    opts = opts_DVFA(;kwargs...)
    DVFA(opts)
end # DVFA(;kwargs...)

"""
    DVFA(opts::opts_DVFA)

Implements a DVFA learner with specified options.

# Examples
```julia-repl
julia> my_opts = opts_DVFA()
julia> DVFA(my_opts)
DVFA
    opts: opts_DVFA
    ...
```
"""
function DVFA(opts::opts_DVFA)
    DVFA(
        opts,                           # opts
        DataConfig(),                   # config
        0.0,                            # threshold_ub
        0.0,                            # threshold_lb
        Array{Int}(undef, 0),           # labels
        Array{Float}(undef, 0, 0),      # W
        Array{Float}(undef, 0),         # M
        Array{Float}(undef, 0),         # T
        0,                              # n_categories
        0,                              # n_clusters
        0                               # epoch
    )
end # DDVFA(opts::opts_DDVFA)

"""
    set_threshold!(art::DVFA)

Configure the threshold values of the DVFA module.
"""
function set_threshold!(art::DVFA)
    # DVFA thresholds
    art.threshold_ub = art.opts.rho_ub * art.config.dim
    art.threshold_lb = art.opts.rho_lb * art.config.dim
end # set_threshold!(art::DVFA)

# Incremental DVFA training method
function train!(art::DVFA, x::RealVector ; y::Integer=0, preprocessed::Bool=false)
    # Flag for if training in supervised mode
    supervised = !iszero(y)

    # Run the sequential initialization procedure
    sample = init_train!(x, art, preprocessed)

    # Initialization
    if isempty(art.W)
        # Set the threshold
        set_threshold!(art)
        # Set the first label as either 1 or the first provided label
        y_hat = supervised ? y : 1
        # Create a new category and cluster
        art.W = ones(art.config.dim_comp, 1)
        art.n_categories = 1
        art.n_clusters = 1
        push!(art.labels, y_hat)
        return y_hat
    end

    # If label is new, break to make new category
    if supervised && !(y in art.labels)
        y_hat = y
        # Update sample labels
        push!(art.labels, y)
        # Fast commit the sample
        art.W = hcat(art.W, sample)
        art.n_categories += 1
        art.n_clusters += 1
        return y_hat
    end

    # Compute the activation and match for all categories
    activation_match!(art, sample)
    # Sort activation function values in descending order
    index = sortperm(art.T, rev=true)

    # Default to mismatch
    mismatch_flag = true
    # Loop over all categories
    for j = 1:art.n_categories
        # Best matching unit
        bmu = index[j]
        # Vigilance test upper bound
        if art.M[bmu] >= art.threshold_ub
            # If supervised and the label differs, trigger mismatch
            if supervised && (art.labels[bmu] != y)
                break
            end
            # Learn the sample
            learn!(art, sample, bmu)
            # Update sample label for output
            # y_hat = supervised ? y : art.labels[bmu]
            y_hat = art.labels[bmu]
            # No mismatch
            mismatch_flag = false
            break
        # Vigilance test lower bound
        elseif art.M[bmu] >= art.threshold_lb
            # If supervised and the label differs, trigger mismatch
            if supervised && (art.labels[bmu] != y)
                break
            end
            # Update sample labels
            y_hat = supervised ? y : art.labels[bmu]
            push!(art.labels, y_hat)
            # Fast commit the sample
            art.W = hcat(art.W, sample)
            art.n_categories += 1
            # No mismatch
            mismatch_flag = false
            break
        end
    end

    # If there was no resonant category, make a new one
    if mismatch_flag
        # Create a new category-to-cluster label
        y_hat = supervised ? y : art.n_clusters + 1
        push!(art.labels, y_hat)
        # Fast commit the sample
        art.W = hcat(art.W, sample)
        # Increment the number of categories and clusters
        art.n_categories += 1
        art.n_clusters += 1
    end

    return y_hat
end # train!(art::DVFA, x::RealVector ; y::Integer=0, preprocessed::Bool=false)

# Incremental DVFA classify method
function classify(art::DVFA, x::RealVector ; preprocessed::Bool=false, get_bmu::Bool=false)
    # Preprocess the data
    sample = init_classify!(x, art, preprocessed)

    # Compute activation and match functions
    activation_match!(art, sample)
    # Sort activation function values in descending order
    index = sortperm(art.T, rev=true)
    mismatch_flag = true
    for jx in 1:art.n_categories
        bmu = index[jx]
        # Vigilance check - pass
        if art.M[bmu] >= art.threshold_ub
            # Current winner
            y_hat = art.labels[bmu]
            mismatch_flag = false
            break
        end
    end

    # If we did not find a resonant category
    if mismatch_flag
        # Create new weight vector
        @debug "Mismatch"
        # Report either the best matching unit or the mismatch label -1
        y_hat = get_bmu ? art.labels[index[1]] : -1
    end

    return y_hat
end # classify(art::DVFA, x::RealVector ; preprocessed::Bool=false, get_bmu::Bool=false)

"""
    activation_match!(art::DVFA, x::RealVector)

Compute and store the activation and match values for the DVFA module.
"""
function activation_match!(art::DVFA, x::RealVector)
    art.T = zeros(art.n_categories)
    art.M = zeros(art.n_categories)
    for jx = 1:art.n_categories
        numerator = norm(element_min(x, art.W[:, jx]), 1)
        art.T[jx] = numerator/(art.opts.alpha + norm(art.W[:, jx], 1))
        art.M[jx] = numerator
    end
end # activation_match!(art::DVFA, x::RealVector)

"""
    learn(art::DVFA, x::RealVector, W::RealVector)

Return the modified weight of the DVFA module conditioned by sample x.
"""
function learn(art::DVFA, x::RealVector, W::RealVector)
    # Update W
    return art.opts.beta .* element_min(x, W) .+ W .* (1 - art.opts.beta)
end # learn(art::DVFA, x::RealVector, W::RealVector)

"""
    learn!(art::DVFA, x::RealVector, index::Integer)

In place learning function.
"""
function learn!(art::DVFA, x::RealVector, index::Integer)
    # Update W
    art.W[:, index] = learn(art, x, art.W[:, index])
end # learn!(art::DVFA, x::RealVector, index::Integer)

"""
    stopping_conditions(art::DVFA)

Stopping conditions for a DVFA module.
"""
function stopping_conditions(art::DVFA)
    return art.epoch >= art.opts.max_epochs
end # stopping_conditions(art::DVFA)
