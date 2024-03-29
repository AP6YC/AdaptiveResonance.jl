"""
    FuzzyART.jl

# Description
Includes all of the structures and logic for running a Gamma-Normalized Fuzzy ART module.

# Authors
- MATLAB implementation: Leonardo Enzo Brito da Silva
- Julia port: Sasha Petrenko <sap625@mst.edu>

# References
1. G. Carpenter, S. Grossberg, and D. Rosen, 'Fuzzy ART: Fast stable learning and categorization of analog patterns by an adaptive resonance system,' Neural Networks, vol. 4, no. 6, pp. 759-771, 1991.
"""

# -----------------------------------------------------------------------------
# TYPES
# -----------------------------------------------------------------------------

"""
Gamma-Normalized Fuzzy ART options struct.

$(_OPTS_DOCSTRING)
"""
@with_kw mutable struct opts_FuzzyART <: ARTOpts @deftype Float
    """
    Vigilance parameter: rho ∈ [0, 1].
    """
    rho = 0.6; @assert rho >= 0.0 && rho <= 1.0

    """
    Choice parameter: alpha > 0.
    """
    alpha = 1e-3; @assert alpha > 0.0

    """
    Learning parameter: beta ∈ (0, 1].
    """
    beta = 1.0; @assert beta > 0.0 && beta <= 1.0

    """
    Pseudo kernel width: gamma >= 1.
    """
    gamma = 3.0; @assert gamma >= 1.0

    """
    Reference gamma for normalization: 0 <= gamma_ref < gamma.
    """
    gamma_ref = 1.0; @assert 0.0 <= gamma_ref && gamma_ref <= gamma

    """
    Maximum number of epochs during training: max_epochs ∈ (1, Inf).
    """
    max_epoch::Int = 1

    """
    Display flag for progress bars.
    """
    display::Bool = false

    """
    Flag to normalize the threshold by the feature dimension.

    **NOTE**: this flag overwrites the `activation` and `match` settings here to their gamma-normalized equivalents along with adjusting the thresold.
    """
    gamma_normalization::Bool = false

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
    match::Symbol = :basic_match

    """
    Selected weight update function.
    """
    update::Symbol = :basic_update
end

"""
Gamma-Normalized Fuzzy ART learner struct

For module options, see [`AdaptiveResonance.opts_FuzzyART`](@ref).

# References
1. G. Carpenter, S. Grossberg, and D. Rosen, 'Fuzzy ART: Fast stable learning and categorization of analog patterns by an adaptive resonance system,' Neural Networks, vol. 4, no. 6, pp. 759-771, 1991.
"""
mutable struct FuzzyART <: AbstractFuzzyART
    """
    FuzzyART options struct.
    """
    opts::opts_FuzzyART

    """
    Data configuration struct.
    """
    config::DataConfig

    """
    Operating module threshold value, a function of the vigilance parameter.
    """
    threshold::Float

    """
    Incremental list of labels corresponding to each F2 node, self-prescribed or supervised.
    """
    labels::ARTVector{Int}

    """
    Activation values for every weight for a given sample.
    """
    T::ARTVector{Float}

    """
    Match values for every weight for a given sample.
    """
    M::ARTVector{Float}

    """
    Category weight matrix.
    """
    W::ARTMatrix{Float}

    """
    Number of weights associated with each category.
    """
    n_instance::ARTVector{Int}

    """
    Number of category weights (F2 nodes).
    """
    n_categories::Int

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
Implements a Fuzzy ART learner with optional keyword arguments.

# Arguments
- `kwargs`: keyword arguments of FuzzyART options (see [`AdaptiveResonance.opts_FuzzyART`](@ref)).

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
    # Create the options from the keyword arguments
    opts = opts_FuzzyART(;kwargs...)

    # Instantiate and return a constructed module
    return FuzzyART(opts)
end

"""
Implements a Fuzzy ART learner with specified options.

# Arguments
- `opts::opts_FuzzyART`: the FuzzyART options struct with specified options (see [`AdaptiveResonance.opts_FuzzyART`](@ref)).

# Examples
```julia-repl
julia> FuzzyART(opts)
FuzzyART
    opts: opts_FuzzyART
    ...
```
"""
function FuzzyART(opts::opts_FuzzyART)
    # Enforce dependent options for gamma normalization
    if opts.gamma_normalization
        opts.activation = :gamma_activation
        opts.match = :gamma_match
    end

    # Construct an empty FuzzyART module
    return FuzzyART(
        opts,                           # opts
        DataConfig(),                   # config
        0.0,                            # threshold
        ARTVector{Int}(undef, 0),       # labels
        ARTVector{Float}(undef, 0),     # T
        ARTVector{Float}(undef, 0),     # M
        ARTMatrix{Float}(undef, 0, 0),  # W
        ARTVector{Int}(undef, 0),       # n_instance
        0,                              # n_categories
        0,                              # epoch
        build_art_stats(),              # stats
    )
end

"""
Create and initialize a FuzzyART with a single sample in one step.

Principally used as a method for initialization within DDVFA.

# Arguments
- `opts::opts_FuzzyART`: the FuzzyART options contains.
- `sample::RealVector`: the sample to use as a basis for setting up the FuzzyART.
- `preprocessed::Bool=false`: flag for if the sample is already complement coded and normalized.
"""
function FuzzyART(opts::opts_FuzzyART, sample::RealVector ; preprocessed::Bool=false)
    # Instantiate the module from the options
    art = FuzzyART(opts)

    # Set up the training dependencies
    init_train!(sample, art, preprocessed)

    # Initialize the module on the first sample
    initialize!(art, sample)

    # Return the constructed and initialized module
    return art
end

# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------

# COMMON DOC: Set threshold function
function set_threshold!(art::FuzzyART)
    # Set the normalized threshold
    if art.opts.gamma_normalization
        art.threshold = art.opts.rho * (art.config.dim ^ art.opts.gamma_ref)
    # Otherwise, vigilance parameter is the threshold
    else
        art.threshold = art.opts.rho
    end
end

# COMMON DOC: create_category! function
function create_category!(art::FuzzyART, x::RealVector, y::Integer)
    # Increment the number of categories
    art.n_categories += 1

    # Increment number of samples associated with new category
    push!(art.n_instance, 1)

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

    # Add the label for the category
    push!(art.labels, y)
end

# COMMON DOC: FuzzyART incremental training method
function train!(art::FuzzyART, x::RealVector ; y::Integer=0, preprocessed::Bool=false)
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

    # If we have a new supervised category, create a new category
    if supervised && !(y in art.labels)
        create_category!(art, sample, y)
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

            # Increment the instance counting
            art.n_instance[bmu] += 1

            # Save the output label for the sample
            y_hat = art.labels[bmu]

            # No mismatch
            mismatch_flag = false
            break
        end
    end

    # If there was no resonant category, make a new one
    if mismatch_flag
        # Keep the bmu as the top activation despite creating a new category
        bmu = index[1]

        # Get the correct label for the new category
        y_hat = supervised ? y : art.n_categories + 1

        # Create a new category
        create_category!(art, sample, y_hat)
    end

    # Update the stored match and activation values
    log_art_stats!(art, bmu, mismatch_flag)

    # Return the training label
    return y_hat
end

# COMMON DOC: FuzzyART incremental classification method
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

    # Iterate over all categories
    for jx in 1:art.n_categories
        # Set the best matching unit
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
        # Report either the best matching unit or the mismatch label -1
        bmu = index[1]

        # Report either the best matching unit or the mismatch label -1
        y_hat = get_bmu ? art.labels[bmu] : -1
    end

    # Update the stored match and activation values
    log_art_stats!(art, bmu, mismatch_flag)

    # Return the inferred label
    return y_hat
end
