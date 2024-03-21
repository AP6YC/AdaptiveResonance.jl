"""
    ART.jl

# Description
Includes all of the unsupervised ART modules definitions.
"""

# -----------------------------------------------------------------------------
# INCLUDES
# -----------------------------------------------------------------------------

# Common code for all ART modules
include("common.jl")

# Single (not distributed) ART modules
include("single/single.jl")

# Distributed ART modules
include("distributed/distributed.jl")

# Convenience constructors of variants of ART modules
include("variants.jl")

# -----------------------------------------------------------------------------
# AGGREGATIONS
# -----------------------------------------------------------------------------

"""
A list of (default) unsupervised ART modules that are available in the `AdaptiveResonance.jl` package.
"""
const ART_MODULES = [
    # Core modules
    FuzzyART,
    DVFA,
    DDVFA,
    # Variants
    GammaNormalizedFuzzyART,
]
