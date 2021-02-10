"""
    opts_FAM()

Implements a Fuzzy ARTMAP learner's options.

# Examples
```julia-repl
julia> my_opts = opts_FAM()
```
"""
@with_kw mutable struct opts_FAM <: AbstractARTOpts @deftype Float64
    # Vigilance parameter: [0, 1]
    rho = 0.6; @assert rho >= 0 && rho <= 1
    # Choice parameter: alpha > 0
    alpha = 1e-7; @assert alpha > 0
    # Match tracking parameter
    epsilon = 1e-3; @assert epsilon > 0 && epsilon < 1
    # Learning parameter: (0, 1]
    beta = 1; @assert beta > 0 && beta <= 1
    # Uncommitted node flag
    uncommitted::Bool = true
    # Display flag
    display::Bool = true

    max_epochs = 1
end

"""
    FAM

Fuzzy ARTMAP struct.
"""
mutable struct FAM <: AbstractART
    opts::opts_FAM
    W::Array{Float64, 2}
    W_old::Array{Float64, 2}
    labels::Array{Int, 1}
    y::Array{Int, 1}
    dim::Int
    n_categories::Int
    epoch::Int
end

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
end

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
        Array{Float64}(undef, 0,0), # W
        Array{Float64}(undef, 0,0), # W_old
        Array{Int}(undef, 0),       # labels
        Array{Int}(undef, 0),       # y
        0,                          # dim
        0,                          # n_categories
        0                           # epoch
    )
end