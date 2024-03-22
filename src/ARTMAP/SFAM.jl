"""
    SFAM.jl

# Description:
Options, structures, and logic for the Simplified Fuzzy ARTMAP (SFAM) module.

# References:
1. G. A. Carpenter, S. Grossberg, N. Markuzon, J. H. Reynolds, and D. B. Rosen, “Fuzzy ARTMAP: A Neural Network Architecture for Incremental Supervised Learning of Analog Multidimensional Maps,” IEEE Trans. Neural Networks, vol. 3, no. 5, pp. 698-713, 1992, doi: 10.1109/72.159059.
"""

# -----------------------------------------------------------------------------
# TYPES
# -----------------------------------------------------------------------------

"""
Implements a Simple Fuzzy ARTMAP learner's options.

$(_OPTS_DOCSTRING)
"""
@with_kw mutable struct opts_SFAM <: ARTOpts @deftype Float
    """
    Vigilance parameter: rho ∈ [0, 1].
    """
    rho = 0.75; @assert rho >= 0.0 && rho <= 1.0

    """
    Choice parameter: alpha > 0.
    """
    alpha = 1e-7; @assert alpha > 0.0

    """
    Match tracking parameter: epsilon ∈ (0, 1).
    """
    epsilon = 1e-3; @assert epsilon > 0.0 && epsilon < 1.0

    """
    Learning parameter: beta ∈ (0, 1].
    """
    beta = 1.0; @assert beta > 0.0 && beta <= 1.0

    """
    Maximum number of epochs during training: max_epoch ∈ [1, Inf).
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
    Selected match function.
    """
    match::Symbol = :basic_match

    """
    Selected activation function.
    """
    activation::Symbol = :basic_activation

    """
    Selected weight update function.
    """
    update::Symbol = :basic_update
end

"""
Simple Fuzzy ARTMAP struct.

For module options, see [`AdaptiveResonance.opts_SFAM`](@ref).

# References
1. G. A. Carpenter, S. Grossberg, N. Markuzon, J. H. Reynolds, and D. B. Rosen, “Fuzzy ARTMAP: A Neural Network Architecture for Incremental Supervised Learning of Analog Multidimensional Maps,” IEEE Trans. Neural Networks, vol. 3, no. 5, pp. 698-713, 1992, doi: 10.1109/72.159059.
"""
mutable struct SFAM <: ARTMAP
    """
    Simplified Fuzzy ARTMAP options struct.
    """
    opts::opts_SFAM

    """
    Data configuration struct.
    """
    config::DataConfig

    """
    Category weight matrix.
    """
    W::ARTMatrix{Float}

    """
    Incremental list of labels corresponding to each F2 node, self-prescribed or supervised.
    """
    labels::ARTVector{Int}

    """
    Number of category weights (F2 nodes).
    """
    n_categories::Int

    """
    Current training epoch.
    """
    epoch::Int

    """
    DDVFA activation values.
    """
    T::ARTVector{Float}

    """
    DDVFA match values.
    """
    M::ARTVector{Float}

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
Implements a Simple Fuzzy ARTMAP learner with optional keyword arguments.

# Arguments
- `kwargs`: keyword arguments to pass to the Simple Fuzzy ARTMAP options struct (see [`AdaptiveResonance.opts_SFAM`](@ref).)

# Examples
By default:
```julia-repl
julia> SFAM()
SFAM
    opts: opts_SFAM
    ...
```

or with keyword arguments:
```julia-repl
julia> SFAM(rho=0.6)
SFAM
    opts: opts_SFAM
    ...
```
"""
function SFAM(;kwargs...)
    opts = opts_SFAM(;kwargs...)
    SFAM(opts)
end

"""
Implements a Simple Fuzzy ARTMAP learner with specified options.

# Arguments
- `opts::opts_SFAM`: the Simple Fuzzy ARTMAP options (see [`AdaptiveResonance.opts_SFAM`](@ref)).

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
        ARTMatrix{Float}(undef, 0, 0),  # W
        ARTVector{Int}(undef, 0),       # labels
        0,                              # n_categories
        0,                              # epoch
        ARTVector{Float}(undef, 0),     # T
        ARTVector{Float}(undef, 0),     # M
        build_art_stats(),              # stats
    )
end

# -----------------------------------------------------------------------------
# ALGORITHMIC METHODS
# -----------------------------------------------------------------------------

# COMMON DOC: SFAM initialization
function initialize!(art::SFAM, x::RealVector, y::Integer)
    # Initialize the weight matrix feature dimension
    art.W = ARTMatrix{Float}(undef, art.config.dim_comp, 0)
    # Create a new category from the sample
    create_category!(art, x, y)
end

# COMMON DOC: SFAM category creation
function create_category!(art::SFAM, x::RealVector, y::Integer)
    # Increment the number of categories
    art.n_categories += 1

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

    # Increment number of samples associated with new category
    # push!(art.n_instance, 1)
    # Add the label for the category
    push!(art.labels, y)
end

# SFAM incremental training method
function train!(art::SFAM, x::RealVector, y::Integer ; preprocessed::Bool=false)
    # Run the sequential initialization procedure
    sample = init_train!(x, art, preprocessed)

    # Initialization
    if isempty(art.W)
        initialize!(art, sample, y)
        return y
    end

    # If we don't have the label, create a new category immediately
    if !(y in art.labels)
        create_category!(art, sample, y)
    # Otherwise, test for a match
    else
        # Baseline vigilance parameter
        rho_baseline = art.opts.rho

        # Compute the activation for all categories
        accommodate_vector!(art.T, art.n_categories)
        for jx in 1:art.n_categories
            art.T[jx] = art_activation(art, sample, jx)
        end

        # Sort activation function values in descending order
        index = sortperm(art.T, rev=true)
        mismatch_flag = true
        accommodate_vector!(art.M, art.n_categories)
        for jx in 1:art.n_categories
            # Set the best-matching-unit index
            bmu = index[jx]
            # Compute match function
            art.M[bmu] = art_match(art, sample, bmu)
            # Current winner
            if art.M[bmu] >= rho_baseline
                if y == art.labels[bmu]
                    # Update the weight and break
                    # art.W[:, index[jx]] = learn(art, sample, art.W[:, index[jx]])
                    learn!(art, sample, bmu)
                    mismatch_flag = false
                    break
                else
                    # Match tracking
                    @debug "Match tracking"
                    rho_baseline = art.M[bmu] + art.opts.epsilon
                end
            end
        end

        # If we triggered a mismatch
        if mismatch_flag
            # Keep the bmu as the top activation despite creating a new category
            bmu = index[1]
            # Create new weight vector
            create_category!(art, sample, y)
        end

        # Update the stored match and activation values
        log_art_stats!(art, bmu, mismatch_flag)
    end

    # ARTMAP guarantees correct training classification, so just return the label
    return y
end

# SFAM incremental classification method
function classify(art::SFAM, x::RealVector ; preprocessed::Bool=false, get_bmu::Bool=false)
    # Run the sequential initialization procedure
    sample = init_classify!(x, art, preprocessed)

    # Compute the activation for all categories
    accommodate_vector!(art.T, art.n_categories)
    for jx in 1:art.n_categories
        art.T[jx] = art_activation(art, sample, jx)
    end

    # Sort activation function values in descending order
    index = sortperm(art.T, rev=true)

    # Default to mismatch
    mismatch_flag = true

    # Iterate over the list of activations
    accommodate_vector!(art.M, art.n_categories)
    for jx in 1:art.n_categories
        # Set the best-matching-unit index
        bmu = index[jx]
        # Compute match function
        art.M[bmu] = art_match(art, sample, bmu)
        # Current winner
        if art.M[bmu] >= art.opts.rho
            y_hat = art.labels[bmu]
            mismatch_flag = false
            break
        end
    end

    # If we did not find a resonant category
    if mismatch_flag
        # Keep the bmu as the top activation
        bmu = index[1]
        # Report either the best matching unit or the mismatch label -1
        y_hat = get_bmu ? art.labels[bmu] : -1
    end

    # Update the stored match and activation values
    log_art_stats!(art, bmu, mismatch_flag)

    return y_hat
end

"""
In-place learning function.
"""
function learn!(art::SFAM, x::RealVector, index::Integer)
    # Compute the updated weight W
    new_vec = art_learn(art, x, index)
    # Replace the weight in place
    replace_mat_index!(art.W, new_vec, index)
    # Return empty
    return
end
