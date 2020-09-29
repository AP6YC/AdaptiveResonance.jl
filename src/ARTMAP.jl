module ARTMAP

using Parameters

export SFAM

include("funcs.jl")

@with_kw mutable struct opts_SFAM @deftype Float64
    # Vigilance parameter: [0, 1]
    rho = 0.6; @assert rho >= 0 && rho <= 1
    # Choice parameter: alpha > 0
    alpha = 1e-3; @assert alpha > 0
    # Learning parameter: (0, 1]
    beta = 1; @assert beta > 0 && beta <= 1
    # Similarity method (activation and match):
    #   'single', 'average', 'complete', 'median', 'weighted', or 'centroid'
    method::String = "single"
    # Display flag
    display::Bool = true
    # shuffle::Bool = false
    random_seed = 1234.5678
end # opts_SFAM

mutable struct SFAM
    opts
    W
    W_old
    labels
    dim
    n_categories
end

function SFAM()
    opts = opts_SFAM()
    SFAM(opts, [], [], [], [], [])
end

function SFAM(opts)
    SFAM(opts, [], [], [], [], [])
end

end