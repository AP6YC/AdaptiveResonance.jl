"""
    common.jl

# Description
Contains all common code for single ART modules (i.e. not distributed models).
"""

# -----------------------------------------------------------------------------
# TYPES
# -----------------------------------------------------------------------------

"""
Abstract supertype of FuzzyART modules.
"""
abstract type AbstractFuzzyART <: ART end

# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------

# COMMON DOC: FuzzyART initialization function
function initialize!(art::AbstractFuzzyART, x::RealVector ; y::Integer=0)
    # Set the threshold
    set_threshold!(art)
    # Initialize the feature dimension of the weights
    art.W = ARTMatrix{Float}(undef, art.config.dim_comp, 0)
    # Set the label to either the supervised label or 1 if unsupervised
    label = !iszero(y) ? y : 1
    # Create a category with the given label
    create_category!(art, x, label)
end
