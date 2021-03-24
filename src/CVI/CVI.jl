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