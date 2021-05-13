"""
    DataConfig

Conatiner to standardize training/testing data configuration.
"""
mutable struct DataConfig
    setup::Bool
    mins::Array{Float64, 1}
    maxs::Array{Float64, 1}
    dim::Int64
    dim_comp::Int64
end # DataConfig

"""
    DataConfig()

Default constructor for a data configuration, not set up.
"""
function DataConfig()
    DataConfig(
        false,                      # setup
        Array{Float64}(undef, 0),   # min
        Array{Float64}(undef, 0),   # max
        0,                          # dim
        0                           # dim_comp
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
    # Get the dimension from one of the arrays
    dim = length(mins)
    # Initialize a Dataconfig with the explicit config
    DataConfig(
        true,   # setup
        mins,   # min
        maxs,    # max
        dim,    # dim
        dim*2   # dim_comp
    )
end # DataConfig(mins::Array, maxs::Array)

"""
    DataConfig(min::Real, max::Real, dim::Int64)

Convenience constructor for DataConfig, requiring only a global min, max, and dim.

This constructor is used in the case that the feature mins and maxs are all the same respectively.
"""
function DataConfig(min::Real, max::Real, dim::Int64)
    DataConfig(
        true,               # setup
        repeat([min], dim), # min
        repeat([max], dim), # max
        dim,                # dim
        dim*2               # dim_comp
    )
end # DataConfig(min::Real, max::Real, dim::Int64)

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
    # conf = confusion_matrix(categorical(y_hat_local), categorical(y_local))
    conf = confusion_matrix(coerce(y_hat_local, OrderedFactor), coerce(y_local, OrderedFactor))
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
        # dim = 1
        # n_samples = length(data)
        dim = length(data)
        n_samples = 1
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
        # n_samples = length(data)
        n_samples = 1
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
    get_data_characteristics(data::Array ; config::DataConfig=DataConfig())

Get the characteristics of the data, taking account if a data config is passed.

If no DataConfig is passed, then the data characteristics come from the array itself. Otherwise, use the config for the statistics of the data and the data array for the number of samples.
"""
function get_data_characteristics(data::Array ; config::DataConfig=DataConfig())
    # If the data is setup, use the config
    if config.setup
        n_samples = get_n_samples(data)
        dim = config.dim
        mins = config.mins
        maxs = config.maxs
    else
        # Get the correct dimensionality and number of samples
        dim, n_samples = get_data_shape(data)
        # Get the ranges for each feature
        mins = [minimum(data[i, :]) for i in 1:dim]
        maxs = [maximum(data[i, :]) for i in 1:dim]
    end
    return dim, n_samples, mins, maxs
end # get_data_characteristics(data::Array ; config::DataConfig=DataConfig())

"""
    linear_normalization(data::Array ; config::DataConfig=DataConfig())

Normalize the data to the range [0, 1] along each feature.
"""
function linear_normalization(data::Array ; config::DataConfig=DataConfig())
    # Get the data characteristics
    dim, n_samples, mins, maxs = get_data_characteristics(data, config=config)

    # Populate a new array with normalized values.
    x_raw = zeros(dim, n_samples)
    for i = 1:dim
        if maxs[i] < mins[i]
            error("Got a data max index that is smaller than the corresonding min")
        elseif maxs[i] - mins[i] != 0
            x_raw[i, :] = (data[i, :] .- mins[i]) ./ (maxs[i] - mins[i])
        end
    end
    return x_raw
end # linear_normalization(data::Array ; config::DataConfig=DataConfig())

"""
    complement_code(data::Array ; config::DataConfig=DataConfig())

Normalize the data x to [0, 1] and returns the augmented vector [x, 1 - x].
"""
function complement_code(data::Array ; config::DataConfig=DataConfig())
    # Normalize the data
    x_raw = linear_normalization(data, config=config)

    # Complement code the data and return a concatenated matrix
    return vcat(x_raw, 1 .- x_raw)
end # complement_code(data::Array ; config::DataConfig=DataConfig())

"""
    get_iterator(opts::O, x::Array) where {O<:AbstractARTOpts}
"""
function get_iterator(opts::O, x::Array) where {O<:AbstractARTOpts}
    # Show a progbar only if the data is 2-D and the option is on
    dim, n_samples = get_data_shape(x)
    single_sample = n_samples == 1

    # Decide if using a progress bar or not
    # Don't use one if either there is a single sample or the display option is off
    prog_bar = single_sample ? false : opts.display

    # Construct the iterator
    iter_raw = 1:n_samples
    iter = prog_bar ?  ProgressBar(iter_raw) : iter_raw

    return iter
end # get_iterator(opts::O, x::Array) where {O<:AbstractARTOpts}

"""
    update_iter(art::A, iter::Union{UnitRange, ProgressBar}, i::Int) where {A<:AbstractART}
"""
function update_iter(art::A, iter::Union{UnitRange, ProgressBar}, i::Int) where {A<:AbstractART}
    # Check explicitly for each, as the function definition restricts the types
    if iter isa ProgressBar
        set_description(iter, string(@sprintf("Ep: %i, ID: %i, Cat: %i", art.epoch, i, art.n_categories)))
    elseif iter isa UnitRange
        return
    end
end # update_iter(art::A, iter::Union{UnitRange, ProgressBar}, i::Int) where {A<:AbstractART}

"""
    get_sample(x::Array, i::Int)

Returns a sample from data array x safely, accounting for 1-D and
"""
function get_sample(x::Array, i::Int)
    # Get the shape of the data, irrespective of data type
    dim, n_samples = get_data_shape(x)
    # Get the type shape of the array
    x_dim = ndims(x)
    # Initialize the sample 1-D array with the original dim
    sample = zeros(dim)
    # Short-circuit error if asking for index out of bounds
    i > n_samples && error("Index of data array out of bounds.")
    # Copy the contents of the input if we got a 1-D array
    if x_dim == 1
        sample = x
    # Otherwise, take the correct slice of the 2-D array
    else
        sample = x[:, i]
    end
    return sample
end # get_sample(x::Array, i::Int)

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
