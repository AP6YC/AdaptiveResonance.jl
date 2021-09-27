"""
    FAM.jl

Description:
    Options, structures, and logic for the Fuzzy ARTMAP (FAM) module.
"""

"""
    opts_FAM()

Implements a Fuzzy ARTMAP learner's options.

# Examples
```julia-repl
julia> my_opts = opts_FAM()
```
"""
@with_kw mutable struct opts_FAM <: ARTOpts @deftype RealFP
    # Vigilance parameter: [0, 1]
    rho = 0.6; @assert rho >= 0.0 && rho <= 1.0
    # Choice parameter: alpha > 0
    alpha = 1e-7; @assert alpha > 0.0
    # Match tracking parameter
    epsilon = 1e-3; @assert epsilon > 0.0 && epsilon < 1.0
    # Learning parameter: (0, 1]
    beta = 1.0; @assert beta > 0.0 && beta <= 1.0
    # Uncommitted node flag
    uncommitted::Bool = true
    # Display flag
    display::Bool = true
    # Maximum number of epochs during training
    max_epochs::Integer = 1
end # opts_FAM()

"""
    FAM <: ARTMAP

Fuzzy ARTMAP struct.
"""
mutable struct FAM <: ARTMAP
    opts::opts_FAM
    config::DataConfig
    W::RealMatrix
    W_old::RealMatrix
    labels::IntegerVector
    y::IntegerVector
    n_categories::Integer
    epoch::Integer
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
        Array{RealFP}(undef, 0,0),  # W
        Array{RealFP}(undef, 0,0),  # W_old
        Array{Integer}(undef, 0),   # labels
        Array{Integer}(undef, 0),   # y
        0,                          # n_categories
        0                           # epoch
    )
end # FAM(opts::opts_FAM)
