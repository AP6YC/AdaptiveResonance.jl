"""
    single.jl

# Description
Aggregates the common code and all the modules of single (i.e. not distributed) ART modules.
"""

# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------
# Single ART module common code
include("common.jl")
# FuzzyART
include("modules/FuzzyART.jl")
# Dual-vigilance FuzzyART
include("modules/DVFA.jl")
