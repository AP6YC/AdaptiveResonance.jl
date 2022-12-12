"""
    DVFA.jl

Description:
    Includes all of the structures and logic for running a Dual-Vigilance Fuzzy ART (DVFA) module.

Authors:
    MATLAB implementation: Leonardo Enzo Brito da Silva
    Julia port: Sasha Petrenko <sap625@mst.edu>

References:
1. L. E. Brito da Silva, I. Elnabarawy and D. C. Wunsch II, 'Dual Vigilance Fuzzy ART,' Neural Networks Letters. To appear.
2. G. Carpenter, S. Grossberg, and D. Rosen, 'Fuzzy ART: Fast stable learning and categorization of analog patterns by an adaptive resonance system,' Neural Networks, vol. 4, no. 6, pp. 759-771, 1991.
"""

"""
Dual Vigilance Fuzzy ART options struct.

$(opts_docstring)
"""
@with_kw mutable struct opts_DVFA <: ARTOpts @deftype Float
    """
    Lower-bound vigilance parameter: rho_lb ∈ [0, 1].
    """
    rho_lb = 0.55; @assert rho_lb >= 0.0 && rho_lb <= 1.0

    """
    Upper bound vigilance parameter: rho_ub ∈ [0, 1].
    """
    rho_ub = 0.75; @assert rho_ub >= 0.0 && rho_ub <= 1.0

    """
    Choice parameter: alpha > 0.
    """
    alpha = 1e-3; @assert alpha > 0.0

    """
    Learning parameter: beta ∈ (0, 1].
    """
    beta = 1.0; @assert beta > 0.0 && beta <= 1.0

    """
    Maximum number of epochs during training.
    """
    max_epochs::Int = 1

    """
    Display flag.
    """
    display::Bool = true
end

"""
Dual Vigilance Fuzzy ARTMAP module struct.

For module options, see [`AdaptiveResonance.opts_DVFA`](@ref).

# References:
1. L. E. Brito da Silva, I. Elnabarawy and D. C. Wunsch II, 'Dual Vigilance Fuzzy ART,' Neural Networks Letters. To appear.
2. G. Carpenter, S. Grossberg, and D. Rosen, 'Fuzzy ART: Fast stable learning and categorization of analog patterns by an adaptive resonance system,' Neural Networks, vol. 4, no. 6, pp. 759-771, 1991.
"""
mutable struct DVFA <: ART
    # Get parameters
    """
    DVFA options struct.
    """
    opts::opts_DVFA

    """
    Data configuration struct.
    """
    config::DataConfig

    # Working variables
    """
    Operating upper bound module threshold value, a function of the upper bound vigilance parameter.
    """
    threshold_ub::Float

    """
    Operating lower bound module threshold value, a function of the lower bound vigilance parameter.
    """
    threshold_lb::Float

    """
    Incremental list of labels corresponding to each F2 node, self-prescribed or supervised.
    """
    labels::ARTVector{Int}

    """
    Category weight matrix.
    """
    W::ARTMatrix{Float}

    """
    Activation values for every weight for a given sample.
    """
    T::ARTVector{Float}

    """
    Match values for every weight for a given sample.
    """
    M::ARTVector{Float}

    """
    Number of category weights (F2 nodes).
    """
    n_categories::Int

    """
    Number of labeled clusters, may be lower than `n_categories`
    """
    n_clusters::Int

    """
    Current training epoch.
    """
    epoch::Int
end

# --------------------------------------------------------------------------- #
# CONSTRUCTORS
# --------------------------------------------------------------------------- #

"""
Implements a DVFA learner with optional keyword arguments.

# Arguments
- `kwargs`: keyword arguments to pass to the DVFA options struct (see [`AdaptiveResonance.opts_DVFA`](@ref).)

# Examples
By default:
```julia-repl
julia> DVFA()
DVFA
    opts: opts_DVFA
    ...
```

or with keyword arguments:
```julia-repl
julia> DVFA(rho=0.7)
DVFA
    opts: opts_DVFA
    ...
```
"""
function DVFA(;kwargs...)
    opts = opts_DVFA(;kwargs...)
    DVFA(opts)
end

"""
Implements a DVFA learner with specified options.

# Arguments
- `opts::opts_DVFA`: the DVFA options (see [`AdaptiveResonance.opts_DVFA`](@ref)).

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
        ARTVector{Int}(undef, 0),       # labels
        ARTMatrix{Float}(undef, 0, 0),  # W
        ARTVector{Float}(undef, 0),     # M
        ARTVector{Float}(undef, 0),     # T
        0,                              # n_categories
        0,                              # n_clusters
        0                               # epoch
    )
end

# --------------------------------------------------------------------------- #
# FUNCTIONS
# --------------------------------------------------------------------------- #

# COMMON DOC: Set threshold function
function set_threshold!(art::DVFA)
    # DVFA thresholds
    art.threshold_ub = art.opts.rho_ub * art.config.dim
    art.threshold_lb = art.opts.rho_lb * art.config.dim
end

"""
Initializes a DVFA learner with an initial sample 'x'.

This function is used during the first training iteraction when the DVFA module is empty.

# Arguments
- `art::DVFA`: the DVFA module to initialize.
- `x::RealVector`: the sample to use for initialization.
- `y::Integer=0`: the optional new label for the first weight of the FuzzyART module. If not specified, defaults the new label to 1.
"""
function initialize!(art::DVFA, x::RealVector ; y::Integer=0)

    # Set the threshold
    set_threshold!(art)

    # Create a new category and cluster
    art.W = ARTMatrix{Float}(undef, art.config.dim_comp, 0)
    # Set the label to either the supervised label or 1 if unsupervised
    label = !iszero(y) ? y : 1

    # append!(art.W, ones(art.config.dim_comp, 1))
    # art.W = ones(art.config.dim_comp, 1)
    append!(art.W, x)

    # Initialize the category counters
    art.n_categories = 1
    art.n_clusters = 1

    push!(art.labels, label)
end

function create_category(art::DVFA, x::RealVector, y::Integer)
    # Increment the number of categories
    art.n_categories += 1
    art.n_clusters += 1
    # Fast commit the sample
    # art.W = hcat(art.W, sample)
    append!(art.W, x)
    # Update sample labels
    push!(art.labels, y)
end

# COMMON DOC: Incremental DVFA training method
function train!(art::DVFA, x::RealVector ; y::Integer=0, preprocessed::Bool=false)
    # Flag for if training in supervised mode
    supervised = !iszero(y)

    # Run the sequential initialization procedure
    sample = init_train!(x, art, preprocessed)

    # Initialization
    if isempty(art.W)
        # Set the first label as either 1 or the first provided label
        y_hat = supervised ? y : 1
        initialize!(art, sample, y=y_hat)
        return y_hat
    end

    # If label is new, break to make new category
    if supervised && !(y in art.labels)
        y_hat = y
        # Update sample labels
        push!(art.labels, y)
        # Fast commit the sample
        # art.W = hcat(art.W, sample)
        append!(art.W, sample)
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
            # art.W = hcat(art.W, sample)
            append!(art.W, sample)
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
        # art.W = hcat(art.W, sample)
        append!(art.W, sample)
        # Increment the number of categories and clusters
        art.n_categories += 1
        art.n_clusters += 1
    end

    return y_hat
end

# COMMON DOC: Incremental DVFA classify method
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
end

"""
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
end

"""
Return the modified weight of the DVFA module conditioned by sample x.
"""
function learn(art::DVFA, x::RealVector, W::RealVector)
    # Update W
    return art.opts.beta .* element_min(x, W) .+ W .* (1 - art.opts.beta)
end

"""
In place learning function.
"""
function learn!(art::DVFA, x::RealVector, index::Integer)
    # Update W
    art.W[:, index] = learn(art, x, art.W[:, index])
end

"""
Stopping conditions for a DVFA module.
"""
function stopping_conditions(art::DVFA)
    return art.epoch >= art.opts.max_epochs
end
