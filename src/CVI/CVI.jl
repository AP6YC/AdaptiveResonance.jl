# include("CONN.jl")
include("XB.jl")
include("DB.jl")
include("PS.jl")

"""
    get_icvi(cvi::T, x::Array{N, 1}, y::M) where {T<:AbstractCVI, N<:Real, M<:Int}


"""
function get_icvi(cvi::T, x::Array{N, 1}, y::M) where {T<:AbstractCVI, N<:Real, M<:Int}
    # Update the ICVI parameters
    param_inc!(cvi, x, y)

    # Compute the criterion value
    evaluate!(cvi)

    # Return that value
    return cvi.criterion_value
end