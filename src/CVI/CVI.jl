# include("CONN.jl")
include("XB.jl")
include("DB.jl")
include("PS.jl")

"""
    get_icvi!(cvi::T, x::Array{N, 1}, y::M) where {T<:AbstractCVI, N<:Real, M<:Int}

Porcelain: update and compute the criterion value incrementally and return it.
"""
function get_icvi!(cvi::T, x::Array{N, 1}, y::M) where {T<:AbstractCVI, N<:Real, M<:Int}
    # Update the ICVI parameters
    param_inc!(cvi, x, y)

    # Compute the criterion value
    evaluate!(cvi)

    # Return that value
    return cvi.criterion_value
end # get_icvi!(cvi::T, x::Array{N, 1}, y::M) where {T<:AbstractCVI, N<:Real, M<:Int}

"""
    get_cvi!(cvi::T, x::Array{N, 2}, y::Array{M, 1}) where {T<:AbstractCVI, N<:Real, M<:Int}

Porcelain: update compute the criterion value in batch and return it.
"""
function get_cvi!(cvi::T, x::Array{N, 2}, y::Array{M, 1}) where {T<:AbstractCVI, N<:Real, M<:Int}
    # Update the CVI parameters in batch
    param_batch!(cvi, x, y)

    # Compute the criterion value
    evaluate!(cvi)

    # Return that value
    return cvi.criterion_value
end # get_cvi!(cvi::T, x::Array{N, 2}, y::Array{M, 1}) where {T<:AbstractCVI, N<:Real, M<:Int}

"""
    expand_array(arr::Array{Float64, 2} ; n::Int64 = 1)

Expand the dimension of a 2-D array by n in each dimension (default 1).
"""
function expand_array(arr::Array{N, 2} ; n::Int64 = 1) where {N<:Real}
    dim, _ = size(arr)
    new_arr = zeros(dim+n, dim+n)
    new_arr[1:end-n, 1:end-n] = arr
    return new_arr
end # expand_array(arr::Array{Float64, 2} ; n::Int64 = 1)

"""
    expand_array(arr::Array{Float64, 1} ; n::Int64 = 1)

Expand a 1-D array by n zeros (default is 1).
"""
function expand_array(arr::Array{N, 1} ; n::Int64 = 1) where {N<:Real}
    dim = length(arr)
    new_arr = zeros(dim+n)
    new_arr[1:end-n] = arr
    return new_arr
end # expand_array(arr::Array{Float64, 1} ; n::Int64 = 1)

# """
#     expand_array!(arr::Array{Float64} ; n::Int64 = 1)

# Expand the dimension of a 1-D or 2-D array in place by n in each dimension (default n).

# Accepts only 1-D or 2-D float arrays.
# Does nothing if n = 0.
# Throws an error if n < 0.
# """
# function expand_array!(arr::Array{Float64} ; n::Int64 = 1)
#     if n==0
#         # If n is zero, make no change
#         return
#     elseif n < 0
#         # Shrinking an array is not defined, so throw an error
#         error("Trying to expand an array to a smaller dimension.")
#     else
#         # Otherwise, expand the array by n dimensions, padded with zeros
#         arr = expand_array(arr ; n=n)
#     end
# end # expand_array!(arr::Array{Float64, 2} ; n::Int64 = 1)

# """
#     expand_array_to!(arr::Array{Float64}, new_dim::Int64 = 0)

# Expand the array in place to new_dim dimensions.

# Accepts 1-D or 2-D float arrays.
# Does nothing if the new dim is smaller than the old one.
# """
# function expand_array_to!(arr::Array{Float64}, new_dim::Int64 = 0)
#     # Get the correct dimensionality and number of samples
#     if ndims(arr) > 1
#         old_dim, _ = size(data)
#     else
#         old_dim = length(data)
#     end

#     # Get the number to expand the dims by
#     n = new_dim - old_dim
#     if n < 1
#         # Do nothing if the new dimension is smaller than the old one
#         return
#     else
#         # Expand the array in place by n dims
#         expand_array!(arr, n=n)
#     end
# end # expand_array_to!(arr::Array{Float64}, new_dim::Int64 = 0)
