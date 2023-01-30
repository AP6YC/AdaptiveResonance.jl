"""
    lib.jl

# Description
Aggregates all common types and functions that are used throughout `AdaptiveResonance.jl`.

# Authors
- Sasha Petrenko <sap625@mst.edu>
"""

# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

# Common docstrings and their templates
include("docstrings.jl")
# Abstract types and constants defining used types
include("conventions.jl")
# Algorithmic common types and functions
include("common.jl")
# Non-algorithmic low-level functions
include("subroutines.jl")
# Activation, match, and update functions and their drivers
include("symbols.jl")
# Common documentation of multiply-dispatched functions
include("common_docs.jl")
