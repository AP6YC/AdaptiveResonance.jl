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

$(OPTS_DOCSTRING)
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
    max_epoch::Int = 1

    """
    Display flag for progress bars.
    """
    display::Bool = false

    """
    Flag to use an uncommitted node when learning.

    If true, new weights are created with ones(dim) and learn on the complement-coded sample.
    If false, fast-committing is used where the new weight is simply the complement-coded sample.
    """
    uncommitted::Bool = false

    """
    Selected activation function.
    """
    activation::Symbol = :basic_activation

    """
    Selected match function.
    """
    match::Symbol = :unnormalized_match

    """
    Selected weight update function.
    """
    update::Symbol = :basic_update
end

"""
Dual Vigilance Fuzzy ARTMAP module struct.

For module options, see [`AdaptiveResonance.opts_DVFA`](@ref).

# References:
1. L. E. Brito da Silva, I. Elnabarawy and D. C. Wunsch II, 'Dual Vigilance Fuzzy ART,' Neural Networks Letters. To appear.
2. G. Carpenter, S. Grossberg, and D. Rosen, 'Fuzzy ART: Fast stable learning and categorization of analog patterns by an adaptive resonance system,' Neural Networks, vol. 4, no. 6, pp. 759-771, 1991.
"""
mutable struct DVFA <: AbstractFuzzyART
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

    """
    Runtime statistics for the module, implemented as a dictionary containing entries at the end of each training iteration.
    These entries include the best-matching unit index and the activation and match values of the winning node.
    """
    stats::ARTStats
end

# -----------------------------------------------------------------------------
# CONSTRUCTORS
# -----------------------------------------------------------------------------

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
        0,                              # epoch
        build_art_stats(),              # stats
    )
end

# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------

# COMMON DOC: Set threshold function
function set_threshold!(art::DVFA)
    # DVFA thresholds
    art.threshold_ub = art.opts.rho_ub * art.config.dim
    art.threshold_lb = art.opts.rho_lb * art.config.dim
end

"""
Creates a new category for the DVFA modules.

# Arguments
- `art::DVFA`: the DVFA module to add a category to.
- `x::RealVector`: the sample to use for adding a category.
- `y::Integer`: the new label for the new category.
"""
function create_category!(art::DVFA, x::RealVector, y::Integer ; new_cluster::Bool=true)
    # Increment the number of categories
    art.n_categories += 1
    # If we are creating a new cluster altogether, increment that
    new_cluster && (art.n_clusters += 1)

    # If we use an uncommitted node
    if art.opts.uncommitted
        # Add a new weight of ones
        append!(art.W, ones(art.config.dim_comp, 1))
        # Learn the uncommitted node on the sample
        learn!(art, x, art.n_categories)
    else
        # Fast commit the sample
        append!(art.W, x)
    end

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
        # Initialize the module with the first sample and label
        initialize!(art, sample, y=y_hat)
        # Return the selected label
        return y_hat
    end

    # If label is new, break to make new category
    if supervised && !(y in art.labels)
        create_category!(art, sample, y)
        return y
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
        # If supervised and the label differs, trigger mismatch
        if supervised && (art.labels[bmu] != y)
            break
        end
        # Vigilance test upper bound
        if art.M[bmu] >= art.threshold_ub
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
            # Update sample labels
            y_hat = supervised ? y : art.labels[bmu]
            # Create a new category in the same cluster
            create_category!(art, sample, y_hat, new_cluster=false)
            # No mismatch
            mismatch_flag = false
            break
        end
    end

    # If there was no resonant category, make a new one
    if mismatch_flag
        # Keep the bmu as the top activation despite creating a new category
        bmu = index[1]
        # Create a new category-to-cluster label
        y_hat = supervised ? y : art.n_clusters + 1
        # Create a new category
        create_category!(art, sample, y_hat)
    end

    # Update the stored match and activation values
    log_art_stats!(art, bmu, mismatch_flag)

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
        bmu = index[1]
        # Report either the best matching unit or the mismatch label -1
        y_hat = get_bmu ? art.labels[bmu] : -1
    end

    # Update the stored match and activation values
    log_art_stats!(art, bmu, mismatch_flag)

    # Return the inferred label
    return y_hat
end
