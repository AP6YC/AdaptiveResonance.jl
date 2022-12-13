"""
    variants.jl

Description:
    Includes convenience constructors for common variants of various ARTMAP modules.
"""

# --------------------------------------------------------------------------- #
# DEFAULT ARTMAP
# --------------------------------------------------------------------------- #

# Shared variant statement for Default ARTMAP
variant_statment = """
Default ARTMAP is a variant of SFAM, usign the [`AdaptiveResonance.opts_SFAM`](@ref) options.
This constructor passes 'choice_by_difference=true' in addition to the keyword argument options you provide
"""

"""
Constructs a Default ARTMAP module using a SFAM module using Default ARTMAP's choice-by-difference activation function.

$(variant_statment)

# Arguments
- `kwargs`: keyword arguments of Simplified FuzzyARTMAP options (see [`AdaptiveResonance.opts_SFAM`](@ref))

# References:
1. G. P. Amis and G. A. Carpenter, 'Default ARTMAP 2,' IEEE Int. Conf. Neural Networks - Conf. Proc., vol. 2, no. September 2007, pp. 777-782, Mar. 2007, doi: 10.1109/IJCNN.2007.4371056.
"""
function DAM(;kwargs...)
    return SFAM(;choice_by_difference=true, kwargs...)
end

"""
Implements a Default ARTMAP module with specified options.

$(variant_statment)

# Arguments
- `opts::opts_SFAM`: the Simplified FuzzyARTMAP options (see [`AdaptiveResonance.opts_SFAM`](@ref)).
"""
function DAM(opts::opts_SFAM)
    return SFAM(opts)
end

"""
Implements a Default ARTMAP module's options.

$(variant_statment)

$(opts_docstring)
"""
function opts_DAM(;kwargs...)
    return opts_SFAM(;choice_by_difference=true, kwargs...)
end
