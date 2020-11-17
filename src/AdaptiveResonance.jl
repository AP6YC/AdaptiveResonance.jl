module AdaptiveResonance

abstract type AbstractARTOpts end
abstract type AbstractART end

include("funcs.jl")
include("ARTMAP/ARTMAP.jl")
include("DDVFA.jl")
include("CVI.jl")

export
# Functions
    train!, classify, performance,
# DDVFA
    DDVFA, opts_DDVFA, GNFA, opts_GNFA,
# ARTMAP
    FAM, opts_FAM,
    DAM, opts_DAM,
    SFAM, opts_SFAM

end # module
