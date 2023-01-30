"""
    distributed.jl

# Description
Aggregates common code and all modules for distributed ART modules, such as DDVFA.
"""

# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

# Common code for distributed ART modules
include("common.jl")
# Distributed Dual-Vigilance FuzzyART
include("modules/DDVFA.jl")
