"""
    variants.jl

Description:
    Includes convenience constructors for common variants of various ART modules.
"""

# --------------------------------------------------------------------------- #
# GAMMA-NORMALIZED FUZZYART
# --------------------------------------------------------------------------- #

# Shared variant statement for GNFA
variant_statement = """
GammaNormalizedFuzzyART is a variant of FuzzyART, using the [`AdaptiveResonance.opts_Fuzzy`](@ref) options.
This constructor passes `gamma_normalization=true` in addition to the keyword argument options you provide.
"""

"""
Constructs a Gamma-Normalized FuzzyART module as a variant of FuzzyART by using the gamma_normalization option.

$(variant_statement)

# Arguments
- `kwargs`: keyword arguments of FuzzyART options (see [`AdaptiveResonance.opts_FuzzyART`](@ref))
"""
function GammaNormalizedFuzzyART(;kwargs...)
    # Return a FuzzyART module with gamma_normalization high in addition to other passed keyword arguments
    return FuzzyART(;gamma_normalization=true, kwargs...)
end

"""
Implements a Gamma-Normalized FuzzyART module with specified options.

$(variant_statement)

# Arguments
- `opts::opts_FuzzyART`: the Fuzzy ART options (see [`AdaptiveResonance.opts_FuzzyART`](@ref)).
"""
function GammaNormalizedFuzzyART(opts::opts_FuzzyART)
    return SFAM(opts)
end

"""
Implements a Gamma-Normalized FuzzyART module's options.

$(variant_statement)

$(opts_docstring)
"""
function opts_GammaNormalizedFuzzyART(;kwargs...)
    return opts_FuzzyART(;choice_by_difference=true, kwargs...)
end
