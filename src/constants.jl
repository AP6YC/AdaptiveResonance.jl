"""
    constants.jl

# Description:
All constant values associated with the package.
"""

"""
A list of all DDVFA linkage methods as strings.
"""
const DDVFA_METHODS = [
    "single",
    "average",
    "complete",
    "median",
    "weighted",
    "centroid"
]
