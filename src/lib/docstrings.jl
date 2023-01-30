"""
    docstrings.jl

# Description
A collection of common docstrings and docstring templates for the package.
"""

# -----------------------------------------------------------------------------
# DOCSTRING TEMPLATES
# -----------------------------------------------------------------------------

# Constants template
@template CONSTANTS =
"""
$(FUNCTIONNAME)

# Description
$(DOCSTRING)
"""

# Types template
@template TYPES =
"""
$(TYPEDEF)

# Summary
$(DOCSTRING)

# Fields
$(TYPEDFIELDS)
"""

# Template for functions, macros, and methods (i.e., constructors)
@template (FUNCTIONS, METHODS, MACROS) =
"""
$(TYPEDSIGNATURES)

# Summary
$(DOCSTRING)

# Method List / Definition Locations
$(METHODLIST)
"""

# -----------------------------------------------------------------------------
# COMMON DOCUMENTATION CONSTANTS
# -----------------------------------------------------------------------------

# Shared options docstring, inserted at the end of `opts_<...>` structs.
const OPTS_DOCSTRING = """
These options are a [`Parameters.jl`](https://github.com/mauro3/Parameters.jl) struct, taking custom options keyword arguments.
Each field has a default value listed below.
"""

const ART_ARG_DOCSTRING = """
- `art::ARTModule`: the ARTModule module.
"""

const X_ARG_DOCSTRING = """
- `x::RealVector`: the sample vector to use.
"""

const W_ARG_DOCSTING = """
- `W::RealVector`: the weight vector to use.
"""

const INDEX_ARG_DOCSTRING = """
- `index::Integer`: the index of the weight column to use.
"""

"""
Shared arguments string for methods using an ART module, sample 'x', and weight vector 'W'.
"""
const ART_X_W_ARGS = """
# Arguments
$(ART_ARG_DOCSTRING)
$(X_ARG_DOCSTRING)
$(W_ARG_DOCSTING)
"""

"""
Shared arguments string for functions updating a column in a matrix.
"""
const MATRIX_REPLACE_ARGS_DOCSTRING = """
# Arguments
- `mat::RealMatrix`: the matrix to update with a replaced column vector.
- `vec::RealVector`: the vector to put in the matrix at the column index.
- `index::Integer`: the column index to put the vector.
"""


# const ART_X_W_ARGS = """
# # Arguments
# - `art::ARTModule`: the ARTModule module.
# - `x::RealVector`: the sample to use.
# - `W::ARTMatrix`: the weight matrix to use.
# - `index::Integer`: the index of the weight to use.
# """
