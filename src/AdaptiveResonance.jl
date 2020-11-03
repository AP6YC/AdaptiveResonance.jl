module AdaptiveResonance

# using Logging
# using Parameters
# using Statistics
# using LinearAlgebra

abstract type AbstractARTOpts end
abstract type AbstractART end

include("basics.jl")
include("funcs.jl")
include("ARTMAP.jl")
include("DDVFA.jl")

export
# Functions
    train!, classify, performance,
# DDVFA
    DDVFA, opts_DDVFA, GNFA, opts_GNFA,
# ARTMAP
    FAM, opts_FAM,
    DAM, opts_DAM,
    SFAM, opts_SFAM,
# Basics
    my_f, foo, doGreet, greet

end # module
