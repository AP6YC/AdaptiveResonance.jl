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

"""
Docstring prefix denoting that the constant is used as a common docstring element for other docstrings.
"""
const _COMMON_DOC = "Common docstring:"

"""
$(_COMMON_DOC) shared options docstring, inserted at the end of `opts_<...>` structs.
"""
const _OPTS_DOCSTRING = """
These options are a [`Parameters.jl`](https://github.com/mauro3/Parameters.jl) struct, taking custom options keyword arguments.
Each field has a default value listed below.
"""

"""
$(_COMMON_DOC) shared argument docstring for ART module arguments.
"""
const _ARG_ART = """
- `art::ARTModule`: the ARTModule module.
"""

"""
$(_COMMON_DOC) shared argument docstring for the input sample of features.
"""
const _ARG_X = """
- `x::RealVector`: the sample vector to use.
"""

"""
$(_COMMON_DOC) shared argument docstring for the weight vector.
"""
const _ARG_W = """
- `W::RealVector`: the weight vector to use.
"""

"""
$(_COMMON_DOC) shared argument docstring for the index of the weight column.
"""
const _ARG_INDEX = """
- `index::Integer`: the index of the weight column to use.
"""

"""
$(_COMMON_DOC) shared arguments string for methods using an ART module, sample 'x', and weight vector 'W'.
"""
const _ARG_ART_X_W = """
# Arguments
$(_ARG_ART)
$(_ARG_X)
$(_ARG_W)
"""

"""
$(_COMMON_DOC) shared arguments string for functions updating a column in a matrix.
"""
const _ARGS_MATRIX_REPLACE = """
# Arguments
- `mat::RealMatrix`: the matrix to update with a replaced column vector.
- `vec::RealVector`: the vector to put in the matrix at the column index.
- `index::Integer`: the column index to put the vector.
"""
