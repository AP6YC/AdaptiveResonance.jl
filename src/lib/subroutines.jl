"""
    subroutines.jl

# Description
Common low-level functions, such as for operating on matrices and vectors.
"""

# -----------------------------------------------------------------------------
# COMMON LOW-LEVEL FUNCTIONS
# -----------------------------------------------------------------------------

"""
Replaces a matrix element with a vector at the column index.

This function dispatches to the low-level replacement strategy.

$_ARGS_MATRIX_REPLACE
"""
function replace_mat_index!(mat::RealMatrix, vec::RealVector, index::Integer)
    unsafe_replace_mat_index!(mat, vec, index)
end

"""
Low-level function for unsafely replacing a matrix column with a given vector.

$_ARGS_MATRIX_REPLACE
"""
function unsafe_replace_mat_index!(mat::RealMatrix, vec::RealVector, index::Integer)
    @inbounds mat[:, index] = vec
end

"""
Extends a vector to a goal length with zeros of its element type to accommodate in-place updates.

# Arguments
- `vec::Vector{T}`: a vector of arbitrary element type.
- `goal_len::Integer`: the length that the vector should be.
"""
function accommodate_vector!(vec::Vector{T}, goal_len::Integer) where {T}
    # While the the vector is not the correct length
    while length(vec) < goal_len
        # Push a zero of the type of the vector elements
        push!(vec, zero(T))
    end
end
