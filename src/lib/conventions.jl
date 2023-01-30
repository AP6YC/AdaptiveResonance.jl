"""
    conventions.jl

# Description
Constants defining conventions of the package along with abstract supertypes.
"""

# -----------------------------------------------------------------------------
# CONSTANTS AND CONVENTIONS
# -----------------------------------------------------------------------------

"""
AdaptiveResonance.jl convention for which 2-D dimension contains the feature dimension.
"""
const ART_DIM = 1

"""
AdaptiveResonance.jl convention for which 2-D dimension contains the number of samples.
"""
const ART_SAMPLES = 2

"""
The type of matrix used by the AdaptiveResonance.jl package, used to configure matrix growth behavior.
"""
const ARTMatrix = ElasticMatrix

"""
The type of vector used by the AdaptiveResonance.jl package, used to configure vector growth behvior.
"""
const ARTVector = Vector

# -----------------------------------------------------------------------------
# ABSTRACT TYPES
# -----------------------------------------------------------------------------

"""
Abstract supertype for all ART module options.
"""
abstract type ARTOpts end               # ART module options

"""
Abstract supertype for both ART (unsupervised) and ARTMAP (supervised) modules.
"""
abstract type ARTModule end             # ART modules

"""
Abstract supertype for all default unsupervised ART modules.
"""
abstract type ART <: ARTModule end      # ART (unsupervised)

"""
Abstract supertype for all supervised ARTMAP modules.
"""
abstract type ARTMAP <: ARTModule end   # ARTMAP (supervised)

"""
Acceptable iterators for ART module training and inference
"""
const ARTIterator = Union{UnitRange, ProgressBar}
