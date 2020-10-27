module AdaptiveResonance

using Logging
using Parameters
using Statistics
using LinearAlgebra

# using CUDAapi
# if has_cuda()		# Check if CUDA is available
#     @info "CUDA is on"
#     import CuArrays		# If CUDA is available, import CuArrays
#     CuArrays.allowscalar(false)
# end

include("basics.jl")
include("funcs.jl")
include("ARTMAP.jl")
include("DDVFA.jl")

# export train!, DDVFA, opts_DDVFA, GNFA, opts_GNFA, SFAM, opts_SFAM, my_f, foo, doGreet, greet

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