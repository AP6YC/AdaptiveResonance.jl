"""
    ARTMAP.jl

Description:
    Includes all of the ARTMAP (i.e., explicitly supervised) ART modules definitions.
"""

include("common.jl")    # train!
include("DAM.jl")       # Default ARTMAP
include("FAM.jl")       # Fuzzy ARTMAP
include("SFAM.jl")      # Simplified Fuzzy ARTMAP
include("ARTSCENE.jl")  # ARTSCENE filters
