"""
    DataConfig

Conatiner to standardize training/testing data configuration.
"""
mutable struct DataConfig
    setup::Bool
    mins::Array{Float64, 1}
    maxs::Array{Float64, 1}
    dim::Int
    dim_comp::Int
end # DataConfig

"""
    DataConfig()

Default constructor for a data configuration, not set up.
"""
function DataConfig()
    DataConfig(false,                       # setup
               Array{Float64}(undef, 0),    # min
               Array{Float64}(undef, 0),    # max
               0,                           # dim
               0                            # dim_comp
    )
end # DataConfig()

"""
    DataConfig(mins::Array, maxs::Array)

Convenience constructor for DataConfig, requiring only mins and maxs of the features.

This constructor is used when the mins and maxs differ across features. The dimension is inferred by the length of the mins and maxs.
"""
function DataConfig(mins::Array, maxs::Array)
    # Verify that the mins and maxs are the same length
    length(mins) != length(maxs) && error("Mins and maxs must be the same length.")

    dim = length(mins)

    DataConfig(true,
               mins,
               max,
               dim,
               dim*2
    )
end # DataConfig(mins::Array, maxs::Array)

"""
    DataConfig(min::Real, max::Real, dim::Int64)

Convenience constructor for DataConfig, requiring only a global min, max, and dim.

This constructor is used in the case that the feature mins and maxs are all the same respectively.
"""
function DataConfig(min::Real, max::Real, dim::Int64)
    DataConfig(true,
               repeat([min], dim),
               repeat([max], dim),
               dim,
               dim*2
    )
end

"""
    element_min(x::Array, W::Array)

Returns the element-wise minimum between sample x and weight W.
"""
function element_min(x::Array, W::Array)
    # Compute the element-wise minimum of two vectors
    return minimum([x W], dims = 2)
end # element_min(x::Array, W::Array)

"""
    performance(y_hat::Array, y::Array)

Returns the categorization performance of y_hat against y.
"""
function performance(y_hat::Array, y::Array)
    # Check lengths
    if length(y_hat) != length(y)
        error("Label vectors must be the same length")
    end

    # Clean up the vectors
    n_mismatch = 0
    y_hat_local = Array{Int64}(undef, 0)
    y_local = Array{Int64}(undef, 0)
    for ix = 1:length(y_hat)
        if y_hat[ix] != -1
            push!(y_hat_local, y_hat[ix])
            push!(y_local, y[ix])
        else
            n_mismatch += 1
        end
    end

    # Compute the confusion matrix and calculate performance as trace/sum
    conf = confusion_matrix(categorical(y_hat_local), categorical(y_local), warn=false)
    return tr(conf.mat)/(sum(conf.mat) + n_mismatch)
end # performance(y_hat::Array, y::Array)

"""
    get_data_shape(data::Array)

Returns the correct feature dimension and number of samples.
"""
function get_data_shape(data::Array)
    # Get the correct dimensionality and number of samples
    if ndims(data) > 1
        dim, n_samples = size(data)
    else
        dim = 1
        n_samples = length(data)
    end

    return dim, n_samples
end # get_data_shape(data::Array)

"""
    get_n_samples(data::Array)

Returns the number of samples, accounting for 1-D and 2-D arrays.
"""
function get_n_samples(data::Array)
    # Get the correct dimensionality and number of samples
    if ndims(data) > 1
        n_samples = size(data)[2]
    else
        n_samples = length(data)
    end

    return n_samples
end # get_n_samples(data::Array)

"""
    data_setup!(config::DataConfig, data::Array)

Sets up the data config for the ART module before training.
"""
function data_setup!(config::DataConfig, data::Array)
    if config.setup
        @warn "Data configuration already set up, overwriting config"
    else
        config.setup = true
    end

    # Get the correct dimensionality and number of samples
    config.dim, n_samples = get_data_shape(data)
    config.dim_comp = 2*config.dim

    # Compute the ranges of each feature
    config.mins = [minimum(data[i, :]) for i in 1:config.dim]
    config.maxs = [maximum(data[i, :]) for i in 1:config.dim]
end # data_setup!(config::DataConfig, data::Array)

"""
    complement_code(data::Array)

Normalize the data x to [0, 1] and returns the augmented vector [x, 1 - x].
"""
function complement_code(data::Array)
    # Get the correct dimensionality and number of samples
    dim, n_samples = get_data_shape(data)

    # Get the ranges for each feature
    mins = [minimum(data[i, :]) for i in 1:dim]
    maxs = [maximum(data[i, :]) for i in 1:dim]

    # Populate a new array with normalized values.
    x_raw = zeros(dim, n_samples)
    for i = 1:dim
        if maxs[i] - mins[i] != 0
            x_raw[i, :] = (data[i, :] .- mins[i]) ./ (maxs[i] - mins[i])
        end
    end

    # Complement code the data and return a concatenated matrix
    return vcat(x_raw, 1 .- x_raw)
end # complement_code(data::Array)

"""
    complement_code(data::Array, config::DataConfig)

Complement code the data based upon the given data config.
"""
function complement_code(data::Array, config::DataConfig)
    # Get the number of points to code
    n_samples = get_n_samples(data)

    # Populate a new array with normalized values.
    x_raw = zeros(config.dim, n_samples)
    for i = 1:config.dim
        if config.maxs[i] - config.mins[i] != 0
            x_raw[i, :] = (data[i, :] .- config.mins[i]) ./ (config.maxs[i] - config.mins[i])
        end
    end

    # Complement code the data and return a concatenated matrix
    return vcat(x_raw, 1 .- x_raw)
end # complement_code(data::Array, config::DataConfig)

# """
#     get_field_meta(obj, field_name)

# Get the value of a struct's field using meta programming.
# """
# function get_field_meta(obj::Any, field_name::String)
#     field = Symbol(field_name)
#     code = quote
#         (obj) -> obj.$field
#     end
#     return eval(code)
# end

# """
#     get_field_native(obj, field_name)

# Get the value of a struct's field through the julia native method.
# """
# function get_field_native(obj::Any, field_name::String)
#     return getfield(obj, Symbol(field_name))
# end

# """
#     similarity_meta(method, F2, field_name, gamma_ref)

# Compute the similarity metric depending on method using meta programming to
# access the correct field.
# """
# function similarity_meta(method::String, F2, field_name::String, gamma_ref::AbstractFloat)
#     @debug "Computing similarity"

#     if field_name != "T" && field_name != "M"
#         error("Incorrect field name for similarity metric.")
#     end

#     field = get_field_native(F2, field_name)

#     # Single linkage
#     if method == "single"
#         value = maximum(field)
#     # Average linkage
#     elseif method == "average"
#         value = mean(field)
#     # Complete linkage
#     elseif method == "complete"
#         value = minimum(field)
#     # Median linkage
#     elseif method == "median"
#         value = median(field)
#     elseif method == "weighted"
#         value = field' * (F2.n / sum(F2.n))
#     elseif method == "centroid"
#         Wc = minimum(F2.W)
#         T = norm(min(sample, Wc), 1)
#         if field_name == "T"
#             value = T
#         elseif field_name == "M"
#             value = (norm(Wc, 1)^gamma_ref)*T
#         end
#     else
#         error("Invalid/unimplemented similarity method")
#     end
#     return value
# end # similarity
