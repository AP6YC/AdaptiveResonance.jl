"""
    ARTMAP.jl

# Description
Includes all of the ARTMAP (i.e., explicitly supervised) ART modules definitions.
"""

# -----------------------------------------------------------------------------
# INCLUDES
# -----------------------------------------------------------------------------

# Common code for all ARTMAP modules, including common dispatches and docstrings
include("common.jl")

# FuzzyARTMAP
include("FAM.jl")

# Simplified Fuzzy ARTMAP
include("SFAM.jl")

# ARTSCENE filters
include("ARTSCENE.jl")

# ARTMAP variants
include("variants.jl")

# -----------------------------------------------------------------------------
# AGGREGATIONS
# -----------------------------------------------------------------------------

"""
A list of supervised ARTMAP modules that are available in the `AdaptiveResonance.jl` package.
"""
const ARTMAP_MODULES = [
    # Core modules
    SFAM,
    # Variants
    DAM,
]
