"""
    GNFA.jl

Description:
    Includes all of the structures and logic for running a Gamma-Normalized Fuzzy ART module.
"""

"""
    opts_GNFA()

Gamma-Normalized Fuzzy ART options struct.

# Examples
```julia-repl
julia> opts_GNFA()
Initialized GNFA
```
"""
@with_kw mutable struct opts_GNFA <: ARTOpts @deftype Float
    # Vigilance parameter: [0, 1]
    rho = 0.6; @assert rho >= 0.0 && rho <= 1.0
    # Choice parameter: alpha > 0
    alpha = 1e-3; @assert alpha > 0.0
    # Learning parameter: (0, 1]
    beta = 1.0; @assert beta > 0.0 && beta <= 1.0
    # "Pseudo" kernel width: gamma >= 1
    gamma = 3.0; @assert gamma >= 1.0
    # gamma = 784; @assert gamma >= 1
    # "Reference" gamma for normalization: 0 <= gamma_ref < gamma
    gamma_ref = 1.0; @assert 0.0 <= gamma_ref && gamma_ref < gamma
    # Similarity method (activation and match):
    #   'single', 'average', 'complete', 'median', 'weighted', or 'centroid'
    method::String = "single"
    # Display flag
    display::Bool = true
    # Maximum number of epochs during training
    max_epochs::Int = 1
end # opts_GNFA

"""
    GNFA <: ART

Gamma-Normalized Fuzzy ART learner struct

# Examples
```julia-repl
julia> GNFA()
GNFA
    opts: opts_GNFA
    ...
```
"""
mutable struct GNFA <: ART
    # Assign numerical parameters from options
    opts::opts_GNFA
    config::DataConfig

    # Working variables
    threshold::Float
    labels::IntegerVector
    T::RealVector
    M::RealVector

    # "Private" working variables
    W::RealMatrix
    W_old::RealMatrix
    n_instance::IntegerVector
    n_categories::Int
    epoch::Int
end # GNFA <: ART

"""
    GNFA()

Implements a Gamma-Normalized Fuzzy ART learner.

# Examples
```julia-repl
julia> GNFA()
GNFA
    opts: opts_GNFA
    ...
```
"""
function GNFA()
    opts = opts_GNFA()
    GNFA(opts)
end # GNFA()

"""
    GNFA(;kwargs...)

Implements a Gamma-Normalized Fuzzy ART learner with keyword arguments.

# Examples
```julia-repl
julia> GNFA(rho=0.7)
GNFA
    opts: opts_GNFA
    ...
```
"""
function GNFA(;kwargs...)
    opts = opts_GNFA(;kwargs...)
    GNFA(opts)
end # GNFA(;kwargs...)

"""
    GNFA(opts::opts_GNFA)

Implements a Gamma-Normalized Fuzzy ART learner with specified options.

# Examples
```julia-repl
julia> GNFA(opts)
GNFA
    opts: opts_GNFA
    ...
```
"""
function GNFA(opts::opts_GNFA)
    GNFA(opts,                          # opts
        DataConfig(),                   # config
        0.0,                            # threshold
        Array{Int}(undef,0),            # labels
        Array{Float}(undef, 0),         # T
        Array{Float}(undef, 0),         # M
        Array{Float}(undef, 0, 0),      # W
        Array{Float}(undef, 0, 0),      # W_old
        Array{Int}(undef, 0),           # n_instance
        0,                              # n_categories
        0                               # epoch
    )
end # GNFA(opts::opts_GNFA)

"""
    GNFA(opts::opts_GNFA, sample::RealVector)

Create and initialize a GNFA with a single sample in one step.
"""
function GNFA(opts::opts_GNFA, sample::RealVector)
    art = GNFA(opts)
    initialize!(art, sample)
    return art
end # GNFA(opts::opts_GNFA, sample::RealVector)

"""
    initialize!(art::GNFA, x::Vector{T} ; y::Integer=0) where {T<:RealFP}

Initializes a GNFA learner with an intial sample 'x'.

# Examples
```julia-repl
julia> my_GNFA = GNFA()
GNFA
    opts: opts_GNFA
    ...
julia> initialize!(my_GNFA, [1 2 3 4])
```
"""
function initialize!(art::GNFA, x::Vector{T} ; y::Integer=0) where {T<:RealFP}
    # Set up the data config
    if art.config.setup
        @warn "Data configuration already set up, overwriting config"
    else
        art.config.setup = true
    end

    # IMPORTANT: Assuming that x is a sample, so each entry is a feature
    dim = length(x)
    art.config.dim_comp = dim
    art.config.dim = Int(dim/2) # Assumes input is already complement coded

    # Initialize the instance and categories counters
    art.n_instance = [1]
    art.n_categories = 1

    # Set the threshold
    art.threshold = art.opts.rho * (art.config.dim^art.opts.gamma_ref)
    # Fast commit the weight
    art.W = Array{T}(undef, art.config.dim_comp, 1)
    # Assign the contents, valid this way for 1-D or 2-D arrays
    art.W[:, 1] = x
    label = y == 0 ? y : 1
    push!(art.labels, label)
end # initialize!(art::GNFA, x::Vector{T} ; y::Integer=0) where {T<:RealFP}

"""
    train!(art::GNFA, x::RealArray ; y::IntegerVector = Vector{Int}())

Trains a GNFA learner with dataset 'x' and optional labels 'y'

# Examples
```julia-repl
julia> my_GNFA = GNFA()
GNFA
    opts: opts_GNFA
    ...
julia> x = load_data()
julia> train!(my_GNFA, x)
```
"""
function train!(art::GNFA, x::RealArray ; y::IntegerVector = Vector{Int}())
    # Flag for if training in supervised mode
    supervised = !isempty(y)
    # Initialization if weights are empty; fast commit the first sample
    if isempty(art.W)
        label = supervised ? y[1] : 1
        push!(art.labels, label)
        initialize!(art, x[:, 1])
        skip_first = true
    else
        skip_first = false
    end

    art.W_old = deepcopy(art.W)

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
            # Skip the first sample if we just initialized
            (i == 1 && skip_first) && continue
            # Grab the sample slice
            sample = get_sample(x, i)
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
                    # Learn the sample
                    learn!(art, sample, bmu)
                    # Update sample labels
                    label = supervised ? y[i] : bmu
                    push!(art.labels, label)
                    # No mismatch
                    mismatch_flag = false
                    break
                end
            end
            # If there was no resonant category, make a new one
            if mismatch_flag
                # Increment the number of categories
                art.n_categories += 1
                # Fast commit
                # art.W = [art.W x[:, i]]
                art.W = hcat(art.W, sample)
                # Increment number of samples associated with new category
                push!(art.n_instance, 1)
                # Update sample labels
                label = supervised ? y[i] : art.n_categories
                push!(art.labels, label)
            end
        end
        # Make sure to start at first sample from now on
        skip_first = false
        # Check for the stopping condition for the whole loop
        if stopping_conditions(art)
            break
        end
        # If we didn't break, deep copy the old weights
        art.W_old = deepcopy(art.W)
    end
end # train!(art::GNFA, x::RealArray ; y::IntegerVector = Vector{Int}())

"""
    classify(art::GNFA, x::RealArray)

Predict categories of 'x' using the GNFA model.

Returns predicted categories 'y_hat'

# Examples
```julia-repl
julia> my_GNFA = GNFA()
GNFA
    opts: opts_GNFA
    ...
julia> x, y = load_data()
julia> train!(my_GNFA, x)
julia> y_hat = classify(my_GNFA, y)
```
"""
function classify(art::GNFA, x::RealArray)
    # Get the number of samples to classify
    n_samples = get_n_samples(x)

    # Initialize the output vector and iterate across all data
    y_hat = zeros(Int, n_samples)
    iter = get_iterator(art.opts, x)
    for ix in iter
        # Update the iterator if necessary
        update_iter(art, iter, ix)
        # Compute activation and match functions
        activation_match!(art, x[:, ix])
        # Sort activation function values in descending order
        index = sortperm(art.T, rev=true)
        mismatch_flag = true
        for jx in 1:art.n_categories
            bmu = index[jx]
            # Vigilance check - pass
            if art.M[bmu] >= art.threshold
                # Current winner
                y_hat[ix] = art.labels[bmu]
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
end # classify(art::GNFA, x::RealArray)

"""
    activation_match!(art::GNFA, x::RealArray)

Computes the activation and match functions of the art module against sample x.

# Examples
```julia-repl
julia> my_GNFA = GNFA()
GNFA
    opts: opts_GNFA
    ...
julia> x, y = load_data()
julia> train!(my_GNFA, x)
julia> x_sample = x[:, 1]
julia> activation_match!(my_GNFA, x_sample)
```
"""
function activation_match!(art::GNFA, x::RealArray)
    art.T = zeros(art.n_categories)
    art.M = zeros(art.n_categories)
    for i = 1:art.n_categories
        W_norm = norm(art.W[:, i], 1)
        art.T[i] = (norm(element_min(x, art.W[:, i]), 1)/(art.opts.alpha + W_norm))^art.opts.gamma
        art.M[i] = (W_norm^art.opts.gamma_ref)*art.T[i]
    end
end # activation_match!(art::GNFA, x::RealArray)

"""
    learn(art::GNFA, x::RealVector, W::RealVector)

Return the modified weight of the art module conditioned by sample x.
"""
function learn(art::GNFA, x::RealVector, W::RealVector)
    # Update W
    return art.opts.beta .* element_min(x, W) .+ W .* (1 - art.opts.beta)
end # learn(art::GNFA, x::RealVector, W::RealVector)

"""
    learn!(art::GNFA, x::RealVector, index::Integer)

In place learning function with instance counting.
"""
function learn!(art::GNFA, x::RealVector, index::Integer)
    # Update W
    art.W[:, index] = learn(art, x, art.W[:, index])
    art.n_instance[index] += 1
end # learn!(art::GNFA, x::RealVector, index::Integer)

"""
    stopping_conditions(art::GNFA)

Stopping conditions for a GNFA module.
"""
function stopping_conditions(art::GNFA)
    return isequal(art.W, art.W_old) || art.epoch >= art.opts.max_epochs
end # stopping_conditions(art::GNFA)