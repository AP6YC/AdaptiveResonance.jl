"""
    ARTMAP.jl

Description:
    Includes all of the ARTMAP (i.e., explicitly supervised) ART modules definitions.
"""

include("common.jl")    # train!
include("FAM.jl")       # Fuzzy ARTMAP
include("SFAM.jl")      # Simplified Fuzzy ARTMAP
include("ARTSCENE.jl")  # ARTSCENE filters
include("variants.jl")  # ARTMAP variants

"""
A list of supervised ARTMAP modules that are available in the `AdaptiveResonance.jl` package.
"""
const ARTMAP_MODULES = [
    # Core modules
    SFAM,
    # Variants
    DAM,
]