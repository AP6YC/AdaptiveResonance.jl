"""
    FAM.jl

Description:
    Options, structures, and logic for the Fuzzy ARTMAP (FAM) module.

References:
[1] G. A. Carpenter, S. Grossberg, N. Markuzon, J. H. Reynolds, and D. B. Rosen, “Fuzzy ARTMAP: A Neural Network Architecture for Incremental Supervised Learning of Analog Multidimensional Maps,” IEEE Trans. Neural Networks, vol. 3, no. 5, pp. 698-713, 1992, doi: 10.1109/72.159059.
"""

"""
    opts_FAM()

Implements a Fuzzy ARTMAP learner's options.

# Keyword Arguments
- `rho::Float`: vigilance value, [0, 1], default 0.6.
- `alpha::Float`: choice parameter, alpha > 0, default 1e-7.
- `epsilon::Float`: match tracking parameter, (0, 1), default 1e-3
- `beta::Float`: learning parameter, (0, 1], default 1.0.
- `uncommitted::Bool`: uncommitted node flag, default true.
- `display::Bool`: display flag, default true.
- `max_epoch::Int`: maximum number of epochs during training, default 1.
"""
@with_kw mutable struct opts_FAM <: ARTOpts @deftype Float
    # Vigilance parameter: [0, 1]
    rho = 0.6; @assert rho >= 0.0 && rho <= 1.0
    # Choice parameter: alpha > 0
    alpha = 1e-7; @assert alpha > 0.0
    # Match tracking parameter: (0, 1)
    epsilon = 1e-3; @assert epsilon > 0.0 && epsilon < 1.0
    # Learning parameter: (0, 1]
    beta = 1.0; @assert beta > 0.0 && beta <= 1.0
    # Uncommitted node flag
    uncommitted::Bool = true
    # Display flag
    display::Bool = true
    # Maximum number of epochs during training
    max_epochs::Int = 1
end # opts_FAM()

"""
    FAM <: ARTMAP

Fuzzy ARTMAP struct.

For module options, see [`AdaptiveResonance.opts_FAM`](@ref).

# References
1. G. A. Carpenter, S. Grossberg, N. Markuzon, J. H. Reynolds, and D. B. Rosen, “Fuzzy ARTMAP: A Neural Network Architecture for Incremental Supervised Learning of Analog Multidimensional Maps,” IEEE Trans. Neural Networks, vol. 3, no. 5, pp. 698-713, 1992, doi: 10.1109/72.159059.
"""
mutable struct FAM <: ARTMAP
    opts::opts_FAM
    config::DataConfig
    W::RealMatrix
    labels::IntegerVector
    n_categories::Int
    epoch::Int
end # FAM <: ARTMAP

"""
    FAM()

Implements a Fuzzy ARTMAP learner.

# Examples
```julia-repl
julia> FAM()
FAM
    opts: opts_FAM
    ...
```
"""
function FAM()
    opts = opts_FAM()
    FAM(opts)
end # FAM()

"""
    FAM(;kwargs...)

Implements a Fuzzy ARTMAP learner with keyword arguments.

# Examples
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
end # FAM(;kwargs...)

"""
    FAM(opts)

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
    FAM(opts,                       # opts_FAM
        DataConfig(),               # config
        Array{Float}(undef, 0,0),   # W
        Array{Int}(undef, 0),       # labels
        0,                          # n_categories
        0                           # epoch
    )
end # FAM(opts::opts_FAM)
