module AdaptiveResonance

# Usings/imports for the whole package declared once
using Parameters
using Logging
using ProgressBars
using Printf
using MLJ: confusion_matrix, categorical
using LinearAlgebra: tr, norm
using Statistics: median, mean

# Abstract types
abstract type AbstractARTOpts end
abstract type AbstractART end

# Include all files
include("funcs.jl")
include("ARTMAP/ARTMAP.jl")
include("ART/ART.jl")
include("CVI/CVI.jl")

# Export all public names
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
