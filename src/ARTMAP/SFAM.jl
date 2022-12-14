"""
    SFAM.jl

# Description:
Options, structures, and logic for the Simplified Fuzzy ARTMAP (SFAM) module.

# References:
1. G. A. Carpenter, S. Grossberg, N. Markuzon, J. H. Reynolds, and D. B. Rosen, “Fuzzy ARTMAP: A Neural Network Architecture for Incremental Supervised Learning of Analog Multidimensional Maps,” IEEE Trans. Neural Networks, vol. 3, no. 5, pp. 698-713, 1992, doi: 10.1109/72.159059.
"""

# --------------------------------------------------------------------------- #
# OPTIONS
# --------------------------------------------------------------------------- #

"""
Implements a Simple Fuzzy ARTMAP learner's options.

$(OPTS_DOCSTRING)
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
    Maximum number of epochs during training: max_epochs ∈ [1, Inf).
    """
    max_epochs::Int = 1

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
end

# --------------------------------------------------------------------------- #
# STRUCTS
# --------------------------------------------------------------------------- #

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
end

# --------------------------------------------------------------------------- #
# CONSTRUCTORS
# --------------------------------------------------------------------------- #

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
        0                               # epoch
    )
end

# --------------------------------------------------------------------------- #
# ALGORITHMIC METHODS
# --------------------------------------------------------------------------- #

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

        # Compute activation function
        T = zeros(art.n_categories)
        for jx in 1:art.n_categories
            T[jx] = art_activation(art, sample, art.W[:, jx])
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
                    # Update the weight and break
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
            create_category!(art, sample, y)
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
        T[jx] = art_activation(art, sample, art.W[:, jx])
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
Stopping conditions for Simple Fuzzy ARTMAP, checked at the end of every epoch.
"""
function stopping_conditions(art::SFAM)
    # Compute the stopping condition, return a bool
    return art.epoch >= art.opts.max_epochs
end

"""
Returns a single updated weight for the Simple Fuzzy ARTMAP module for weight
vector W and sample x.
"""
function learn(art::SFAM, x::RealVector, W::RealVector)
    # Update W
    return art.opts.beta .* element_min(x, W) .+ W .* (1 - art.opts.beta)
end

"""
In-place learning function.
"""
function learn!(art::SFAM, x::RealVector, index::Integer)
    # Update W at the index
    art.W[:, index] = learn(art, x, art.W[:, index])
end

"""
Returns the match function for the Simple Fuzzy ARTMAP module with weight W and
sample x.
"""
function art_match(art::SFAM, x::RealVector, W::RealVector)
    # Compute M and return
    return norm(element_min(x, W), 1) / art.config.dim
end
