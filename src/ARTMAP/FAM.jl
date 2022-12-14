"""
    FAM.jl

# Description:
Options, structures, and logic for the Fuzzy ARTMAP (FAM) module.

# References:
1. G. A. Carpenter, S. Grossberg, N. Markuzon, J. H. Reynolds, and D. B. Rosen, “Fuzzy ARTMAP: A Neural Network Architecture for Incremental Supervised Learning of Analog Multidimensional Maps,” IEEE Trans. Neural Networks, vol. 3, no. 5, pp. 698-713, 1992, doi: 10.1109/72.159059.
"""

# --------------------------------------------------------------------------- #
# OPTIONS
# --------------------------------------------------------------------------- #

"""
Implements a Fuzzy ARTMAP learner's options.

$(OPTS_DOCSTRING)
"""
@with_kw mutable struct opts_FAM <: ARTOpts @deftype Float
    """
    Vigilance parameter: rho ∈ [0, 1].
    """
    rho = 0.6; @assert rho >= 0.0 && rho <= 1.0

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
    Maximum number of epochs during training: max_epochs ∈ [1, Inf)
    """
    max_epochs::Int = 1

    """
    Uncommitted node flag.
    """
    uncommitted::Bool = true

    """
    Display flag for progress bars.
    """
    display::Bool = false
end

# --------------------------------------------------------------------------- #
# STRUCTS
# --------------------------------------------------------------------------- #

"""
Fuzzy ARTMAP struct.

For module options, see [`AdaptiveResonance.opts_FAM`](@ref).

# References
1. G. A. Carpenter, S. Grossberg, N. Markuzon, J. H. Reynolds, and D. B. Rosen, “Fuzzy ARTMAP: A Neural Network Architecture for Incremental Supervised Learning of Analog Multidimensional Maps,” IEEE Trans. Neural Networks, vol. 3, no. 5, pp. 698-713, 1992, doi: 10.1109/72.159059.
"""
mutable struct FAM <: ARTMAP
    """
    Fuzzy ARTMAP options struct.
    """
    opts::opts_FAM

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
Implements a Fuzzy ARTMAP learner with optional keyword arguments.

# Examples
By default:
```julia-repl
julia> FAM()
FAM
    opts: opts_FAM
    ...
```

or with keyword arguments:
```julia-repl
julia> FAM(rho=0.7)
FAM
    opts: opts_FAM
    ...
```
"""
function FAM(;kwargs...)
    opts = opts_FAM(;kwargs...)
    FAM(opts)
end

"""
Implements a Fuzzy ARTMAP learner with specified options.

# Examples
```julia-repl
julia> opts = opts_FAM()
julia> FAM(opts)
FAM
    opts: opts_FAM
    ...
```
"""
function FAM(opts::opts_FAM)
    FAM(
        opts,                           # opts_FAM
        DataConfig(),                   # config
        ARTMatrix{Float}(undef, 0, 0),  # W
        ARTVector{Int}(undef, 0),       # labels
        0,                              # n_categories
        0                               # epoch
    )
end
