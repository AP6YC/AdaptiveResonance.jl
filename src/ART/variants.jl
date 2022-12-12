"""
    variants.jl

Description:
    Includes convenience constructors for common variants of various ART modules.
"""

"""
Constructs a Gamma-Normalized FuzzyART module by simply using the gamma_normalization option of the FuzzyART module.
"""
function GammaNormalizedFuzzyART(;kwargs...)
    return FuzzyART(;gamma_normalization=true, kwargs...)
end
