"""
    ART.jl

Description:
    Includes all of the unsupervised ART modules definitions.
"""

include("common.jl")    # train!, classify
include("FuzzyART.jl")  # FuzzyART
include("DDVFA.jl")     # DDVFA
include("DVFA.jl")      # DVFA

"""
A list of (default) unsupervised ART modules that are available in the `AdaptiveResonance.jl` package.
"""
const ART_MODULES = [
    FuzzyART,
    DVFA,
    DDVFA,
]