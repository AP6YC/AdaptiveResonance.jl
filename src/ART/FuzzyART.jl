"""
    FuzzyART.jl

Description:
    Includes all of the structures and logic for running a Gamma-Normalized Fuzzy ART module.
"""

# --------------------------------------------------------------------------- #
# OPTIONS
# --------------------------------------------------------------------------- #

"""
    opts_FuzzyART()

Gamma-Normalized Fuzzy ART options struct.

# Examples
```julia-repl
julia> opts_FuzzyART()
Initialized FuzzyART
```
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
    threshold_normalization::Bool = true
end # opts_FuzzyART

# --------------------------------------------------------------------------- #
# STRUCTS
# --------------------------------------------------------------------------- #

"""
    FuzzyART <: ART

Gamma-Normalized Fuzzy ART learner struct

# Examples
```julia-repl
julia> FuzzyART()
FuzzyART
    opts: opts_FuzzyART
    ...
```
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
    FuzzyART()

Implements a Gamma-Normalized Fuzzy ART learner.

# Examples
```julia-repl
julia> FuzzyART()
FuzzyART
    opts: opts_FuzzyART
    ...
```
"""
function FuzzyART()
    opts = opts_FuzzyART()
    FuzzyART(opts)
end # FuzzyART()

"""
    FuzzyART(;kwargs...)

Implements a Gamma-Normalized Fuzzy ART learner with keyword arguments.

# Examples
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
    if art.opts.threshold_normalization
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

"""
    train!(art::FuzzyART, x::RealVector ; y::Integer = 0, preprocessed::Bool=false)
"""
function train!(art::FuzzyART, x::RealVector ; y::Integer = 0, preprocessed::Bool=false)
    # Flag for if training in supervised mode
    supervised = !iszero(y)

    # Run the sequential initialization procedure
    x = init_train!(x, art, preprocessed)

    # Initialization if weights are empty; fast commit the first sample
    if isempty(art.W)
        label = supervised ? y : 1
        initialize!(art, x, y=label)
        return
    end

    # Compute activation/match functions
    activation_match!(art, x)
    # Sort activation function values in descending order
    index = sortperm(art.T, rev=true)
    # Initialize mismatch as true
    mismatch_flag = true

    # If we have a new supervised category, create a new category
    if supervised && !(y in art.labels)
        create_category(art, x, y)
        return y
    end
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
            learn!(art, x, bmu)
            # No mismatch
            mismatch_flag = false
            break
        end
    end
    # If there was no resonant category, make a new one
    if mismatch_flag
        # Get the correct label for the new category
        label = supervised ? y : art.n_categories + 1
        # Create a new category
        create_category(art, x, label)
    end

    return y
end # train!(art::FuzzyART, x::RealVector ; y::Integer = 0, preprocessed::Bool=false)

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

"""
    train!(art::FuzzyART, x::RealMatrix ; y::IntegerVector = Vector{Int}())

Trains a FuzzyART learner with dataset 'x' and optional labels 'y'

# Examples
```julia-repl
julia> my_FuzzyART = FuzzyART()
FuzzyART
    opts: opts_FuzzyART
    ...
julia> x = load_data()
julia> train!(my_FuzzyART, x)
```
"""
function train!(art::FuzzyART, x::RealMatrix ; y::IntegerVector = Vector{Int}(), preprocessed::Bool=false)
    # Flag for if training in supervised mode
    supervised = !isempty(y)

    # Run the batch initialization procedure
    x = init_train!(x, art, preprocessed)

    # Learning
    art.epoch = 0
    while true
        # Increment the epoch and get the iterator
        art.epoch += 1
        iter = get_iterator(art.opts, x)
        # Loop over samples
        for i = iter
            # Update the iterator if necessary
            update_iter(art, iter, i)
            # Grab the sample slice
            sample = get_sample(x, i)
            # sample = x[:, i]
            # Train on the sample
            local_y = supervised ? y[i] : 0
            train!(art, sample, y=local_y, preprocessed=true)
        end
        # Check for the stopping condition for the whole loop
        if stopping_conditions(art)
            break
        end
    end
end # train!(art::FuzzyART, x::RealMatrix ; y::IntegerVector = Vector{Int}())

"""
    classify(art::FuzzyART, x::RealVector ; preprocessed::Bool=false, get_bmu::Bool=false)
"""
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
end

"""
    classify(art::FuzzyART, x::RealMatrix ; preprocessed::Bool=false, get_bmu::Bool=false)

Batch predict categories of 'x' using the FuzzyART model.

Returns predicted categories 'y_hat'

# Examples
```julia-repl
julia> my_FuzzyART = FuzzyART()
FuzzyART
    opts: opts_FuzzyART
    ...
julia> x, y = load_data()
julia> train!(my_FuzzyART, x)
julia> y_hat = classify(my_FuzzyART, y)
```
"""
function classify(art::FuzzyART, x::RealMatrix ; preprocessed::Bool=false, get_bmu::Bool=false)
    # Preprocess the data
    x = init_classify!(x, art, preprocessed)
    # Get the number of samples to classify
    n_samples = get_n_samples(x)

    # Initialize the output vector and iterate across all data
    y_hat = zeros(Int, n_samples)
    iter = get_iterator(art.opts, x)
    for ix in iter
        # Update the iterator if necessary
        update_iter(art, iter, ix)
        sample = x[:, ix]
        y_hat[ix] = classify(art, sample, preprocessed=true, get_bmu=get_bmu)
    end
    return y_hat
end # classify(art::FuzzyART, x::RealMatrix ; preprocessed::Bool=false, get_bmu::Bool=false)

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
        art.T[i] = (norm(element_min(x, art.W[:, i]), 1)/(art.opts.alpha + W_norm))^art.opts.gamma
        if art.opts.threshold_normalization
            art.M[i] = (W_norm^art.opts.gamma_ref)*art.T[i]
        else
            art.M[i] = ((W_norm/norm(x, 1))^art.opts.gamma_ref)*art.T[i]
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
