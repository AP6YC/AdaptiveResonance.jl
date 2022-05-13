"""
    FuzzyART.jl

Description:
    Includes all of the structures and logic for running a Gamma-Normalized Fuzzy ART module.

References:
[1] G. Carpenter, S. Grossberg, and D. Rosen, "Fuzzy ART: Fast stable learning and categorization of analog patterns by an adaptive resonance system," Neural Networks, vol. 4, no. 6, pp. 759-771, 1991.
"""

# --------------------------------------------------------------------------- #
# OPTIONS
# --------------------------------------------------------------------------- #

"""
    opts_FuzzyART(;kwargs)

Gamma-Normalized Fuzzy ART options struct.

# Keyword Arguments
- `rho::Float`: vigilance value, [0, 1], default 0.6.
- `alpha::Float`: choice parameter, alpha > 0, default 1e-3.
- `beta::Float`: learning parameter, (0, 1], default 1.0.
- `gamma::Float`: "pseudo" kernel width, gamma >= 1, default 3.0.
- `gamma_ref::Float`: "reference" kernel width, 0 <= gamma_ref < gamma, default 1.0.
- `display::Bool`: display flag, default true.
- `max_epoch::Int`: maximum number of epochs during training, default 1.
- `gamma_normalization::Bool`: normalize the threshold by the feature dimension, default false.
"""
@with_kw mutable struct opts_FuzzyART <: ARTOpts @deftype Float
    # Vigilance parameter: [0, 1]
    rho = 0.6; @assert rho >= 0.0 && rho <= 1.0
    # Choice parameter: alpha > 0
    alpha = 1e-3; @assert alpha > 0.0
    # Learning parameter: (0, 1]
    beta = 1.0; @assert beta > 0.0 && beta <= 1.0
    # "Pseudo" kernel width: gamma >= 1
    gamma = 3.0; @assert gamma >= 1.0
    # "Reference" gamma for normalization: 0 <= gamma_ref < gamma
    gamma_ref = 1.0; @assert 0.0 <= gamma_ref && gamma_ref <= gamma
    # Display flag
    display::Bool = true
    # Maximum number of epochs during training
    max_epochs::Int = 1
    # Normalize the threshold by the feature dimension
    gamma_normalization::Bool = false
end # opts_FuzzyART

# --------------------------------------------------------------------------- #
# STRUCTS
# --------------------------------------------------------------------------- #

"""
    FuzzyART <: ART

Gamma-Normalized Fuzzy ART learner struct

For module options, see [`AdaptiveResonance.opts_FuzzyART`](@ref).

# Option Parameters
- `opts::opts_FuzzyART`: FuzzyART options struct.
- `config::DataConfig`: data configuration struct.

# Working Parameters
- `threshold::Float`: operating module threshold value, a function of the vigilance parameter.
- `labels::IntegerVector`: incremental list of labels corresponding to each F2 node, self-prescribed or supervised.
- `T::RealVector`: activation values for every weight for a given sample.
- `M::RealVector`: match values for every weight for a given sample.
- `W::RealMatrix`: category weight matrix.
- `n_instance::IntegerVector`: number of weights associated with each category.
- `n_categories::Int`: number of category weights (F2 nodes).
- `epoch::Int`: current training epoch.

# References
1. G. Carpenter, S. Grossberg, and D. Rosen, "Fuzzy ART: Fast stable learning and categorization of analog patterns by an adaptive resonance system," Neural Networks, vol. 4, no. 6, pp. 759-771, 1991.
"""
mutable struct FuzzyART <: ART
    # Assign numerical parameters from options
    opts::opts_FuzzyART
    config::DataConfig

    # Working variables
    threshold::Float
    labels::IntegerVector
    T::RealVector
    M::RealVector

    # "Private" working variables
    W::RealMatrix
    n_instance::IntegerVector
    n_categories::Int
    epoch::Int
end # FuzzyART <: ART

# --------------------------------------------------------------------------- #
# CONSTRUCTORS
# --------------------------------------------------------------------------- #

"""
    FuzzyART(;kwargs...)

Implements a Gamma-Normalized Fuzzy ART learner with optional keyword arguments.

# Examples
By default:
```julia-repl
julia> FuzzyART()
FuzzyART
    opts: opts_FuzzyART
    ...
```

or with keyword arguments:
```julia-repl
julia> FuzzyART(rho=0.7)
FuzzyART
    opts: opts_FuzzyART
    ...
```
"""
function FuzzyART(;kwargs...)
    opts = opts_FuzzyART(;kwargs...)
    FuzzyART(opts)
end # FuzzyART(;kwargs...)

"""
    FuzzyART(opts::opts_FuzzyART)

Implements a Gamma-Normalized Fuzzy ART learner with specified options.

# Examples
```julia-repl
julia> FuzzyART(opts)
FuzzyART
    opts: opts_FuzzyART
    ...
```
"""
function FuzzyART(opts::opts_FuzzyART)
    FuzzyART(opts,                          # opts
        DataConfig(),                   # config
        0.0,                            # threshold
        Array{Int}(undef,0),            # labels
        Array{Float}(undef, 0),         # T
        Array{Float}(undef, 0),         # M
        Array{Float}(undef, 0, 0),      # W
        Array{Int}(undef, 0),           # n_instance
        0,                              # n_categories
        0                               # epoch
    )
end # FuzzyART(opts::opts_FuzzyART)

"""
    FuzzyART(opts::opts_FuzzyART, sample::RealVector)

Create and initialize a FuzzyART with a single sample in one step.
"""
function FuzzyART(opts::opts_FuzzyART, sample::RealVector ; preprocessed::Bool=false)
    art = FuzzyART(opts)
    init_train!(sample, art, preprocessed)
    initialize!(art, sample)
    return art
end # FuzzyART(opts::opts_FuzzyART, sample::RealVector)

# --------------------------------------------------------------------------- #
# ALGORITHMIC METHODS
# --------------------------------------------------------------------------- #

function set_threshold!(art::FuzzyART)
    if art.opts.gamma_normalization
        art.threshold = art.opts.rho*(art.config.dim^art.opts.gamma_ref)
    else
        art.threshold = art.opts.rho
    end
end # set_threshold!(art::FuzzyART)

"""
    initialize!(art::FuzzyART, x::Vector{T} ; y::Integer=0) where {T<:RealFP}

Initializes a FuzzyART learner with an intial sample 'x'.

# Examples
```julia-repl
julia> my_FuzzyART = FuzzyART()
FuzzyART
    opts: opts_FuzzyART
    ...
julia> initialize!(my_FuzzyART, [1 2 3 4])
```
"""
function initialize!(art::FuzzyART, x::Vector{T} ; y::Integer=0) where {T<:RealFP}
    # Initialize the instance and categories counters
    art.n_instance = [1]
    art.n_categories = 1

    # Set the threshold
    set_threshold!(art)

    # Fast commit the weight
    art.W = Array{T}(undef, art.config.dim_comp, 1)

    # Assign the contents, valid this way for 1-D or 2-D arrays
    art.W[:, 1] = x

    # Set the label to either the supervised label or 1 if unsupervised
    label = !iszero(y) ? y : 1

    # Add the label to the label list
    push!(art.labels, label)
end # initialize!(art::FuzzyART, x::Vector{T} ; y::Integer=0) where {T<:RealFP}

# FuzzyART incremental training method
function train!(art::FuzzyART, x::RealVector ; y::Integer=0, preprocessed::Bool=false)
    # Flag for if training in supervised mode
    supervised = !iszero(y)

    # Run the sequential initialization procedure
    sample = init_train!(x, art, preprocessed)

    # Initialization if weights are empty; fast commit the first sample
    if isempty(art.W)
        y_hat = supervised ? y : 1
        initialize!(art, sample, y=y_hat)
        return y_hat
    end

    # If we have a new supervised category, create a new category
    if supervised && !(y in art.labels)
        create_category(art, sample, y)
        return y
    end

    # Compute activation/match functions
    activation_match!(art, sample)
    # Sort activation function values in descending order
    index = sortperm(art.T, rev=true)
    # Initialize mismatch as true
    mismatch_flag = true

    # Loop over all categories
    for j = 1:art.n_categories
        # Best matching unit
        bmu = index[j]
        # Vigilance check - pass
        if art.M[bmu] >= art.threshold
            # If supervised and the label differed, force mismatch
            if supervised && (art.labels[bmu] != y)
                break
            end
            # Learn the sample
            learn!(art, sample, bmu)
            # Save the output label for the sample
            y_hat = art.labels[bmu]
            # No mismatch
            mismatch_flag = false
            break
        end
    end

    # If there was no resonant category, make a new one
    if mismatch_flag
        # Get the correct label for the new category
        y_hat = supervised ? y : art.n_categories + 1
        # Create a new category
        create_category(art, sample, y_hat)
    end

    return y_hat
end # train!(art::FuzzyART, x::RealVector ; y::Integer=0, preprocessed::Bool=false)

"""
    create_category(art::FuzzyART, x::RealVector, y::Integer)
"""
function create_category(art::FuzzyART, x::RealVector, y::Integer)
    # Increment the number of categories
    art.n_categories += 1
    # Fast commit
    art.W = hcat(art.W, x)
    # Increment number of samples associated with new category
    push!(art.n_instance, 1)
    # Add the label for the ategory
    push!(art.labels, y)
end # create_category(art::FuzzyART, x::RealVector, y::Integer)

# FuzzyART incremental classification method
function classify(art::FuzzyART, x::RealVector ; preprocessed::Bool=false, get_bmu::Bool=false)
    # Preprocess the data
    x = init_classify!(x, art, preprocessed)

    # Compute activation and match functions
    activation_match!(art, x)
    # Sort activation function values in descending order
    index = sortperm(art.T, rev=true)
    # Default is mismatch
    mismatch_flag = true
    y_hat = -1
    for jx in 1:art.n_categories
        bmu = index[jx]
        # Vigilance check - pass
        if art.M[bmu] >= art.threshold
            # Current winner
            y_hat = art.labels[bmu]
            mismatch_flag = false
            break
        end
    end
    # If we did not find a match
    if mismatch_flag
        # Create new weight vector
        @debug "Mismatch"
        # Report either the best matching unit or the mismatch label -1
        y_hat = get_bmu ? art.labels[index[1]] : -1
    end
    return y_hat
end # classify(art::FuzzyART, x::RealVector ; preprocessed::Bool=false, get_bmu::Bool=false)

"""
    activation_match!(art::FuzzyART, x::RealVector)

Computes the activation and match functions of the art module against sample x.

# Examples
```julia-repl
julia> my_FuzzyART = FuzzyART()
FuzzyART
    opts: opts_FuzzyART
    ...
julia> x, y = load_data()
julia> train!(my_FuzzyART, x)
julia> x_sample = x[:, 1]
julia> activation_match!(my_FuzzyART, x_sample)
```
"""
function activation_match!(art::FuzzyART, x::RealVector)
    art.T = zeros(art.n_categories)
    art.M = zeros(art.n_categories)
    for i = 1:art.n_categories
        W_norm = norm(art.W[:, i], 1)
        numerator = norm(element_min(x, art.W[:, i]), 1)
        art.T[i] = (numerator/(art.opts.alpha + W_norm))^art.opts.gamma
        if art.opts.gamma_normalization
            art.M[i] = (W_norm^art.opts.gamma_ref)*art.T[i]
        else
            art.M[i] = numerator/norm(x, 1)
        end
    end
end # activation_match!(art::FuzzyART, x::RealVector)

"""
    learn(art::FuzzyART, x::RealVector, W::RealVector)

Return the modified weight of the art module conditioned by sample x.
"""
function learn(art::FuzzyART, x::RealVector, W::RealVector)
    # Update W
    return art.opts.beta .* element_min(x, W) .+ W .* (1 - art.opts.beta)
end # learn(art::FuzzyART, x::RealVector, W::RealVector)

"""
    learn!(art::FuzzyART, x::RealVector, index::Integer)

In place learning function with instance counting.
"""
function learn!(art::FuzzyART, x::RealVector, index::Integer)
    # Update W
    art.W[:, index] = learn(art, x, art.W[:, index])
    art.n_instance[index] += 1
end # learn!(art::FuzzyART, x::RealVector, index::Integer)

"""
    stopping_conditions(art::FuzzyART)

Stopping conditions for a FuzzyART module.
"""
function stopping_conditions(art::FuzzyART)
    return art.epoch >= art.opts.max_epochs
end # stopping_conditions(art::FuzzyART)
