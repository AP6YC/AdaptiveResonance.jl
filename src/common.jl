# -------------------------------------------
# Document: common.jl
# Author: Sasha Petrenko <sap625@mst.edu>
# Description:
#   Types and functions that are used throughout AdaptiveResonance.jl.
# -------------------------------------------

# -------------------------------------------
# Aliases
# -------------------------------------------
#   **Taken from StatsBase.jl**
#
#  These types signficantly reduces the need of using
#  type parameters in functions (which are often just
#  for the purpose of restricting the arrays to real)
#
# These could be removed when the Base supports
# covariant type notation, i.e. AbstractVector{<:Real}

# Real-numbered aliases
const RealArray{T<:Real, N} = AbstractArray{T, N}
const RealVector{T<:Real} = AbstractArray{T, 1}
const RealMatrix{T<:Real} = AbstractArray{T, 2}

# Integered aliases
const IntegerArray{T<:Integer, N} = AbstractArray{T, N}
const IntegerVector{T<:Integer} = AbstractArray{T, 1}
const IntegerMatrix{T<:Integer} = AbstractArray{T, 2}

# Specifically floating-point aliases
const RealFP = Union{Float32, Float64}

# System's largest native floating point variable
const Float = (Sys.WORD_SIZE == 64 ? Float64 : Float32)

# Acceptable iterators for ART module training and inference
const ARTIterator = Union{UnitRange, ProgressBar}

"""
    DataConfig

Conatiner to standardize training/testing data configuration.
"""
mutable struct DataConfig
    setup::Bool
    mins::RealVector
    maxs::RealVector
    dim::Int
    dim_comp::Int
end # DataConfig

"""
    DataConfig()

Default constructor for a data configuration, not set up.
"""
function DataConfig()
    DataConfig(
        false,                      # setup
        Array{Float}(undef, 0),     # min
        Array{Float}(undef, 0),     # max
        0,                          # dim
        0                           # dim_comp
    )
end # DataConfig()

"""
    DataConfig(mins::RealVector, maxs::RealVector)

Convenience constructor for DataConfig, requiring only mins and maxs of the features.

This constructor is used when the mins and maxs differ across features. The dimension is inferred by the length of the mins and maxs.
"""
function DataConfig(mins::RealVector, maxs::RealVector)
    # Verify that the mins and maxs are the same length
    length(mins) != length(maxs) && error("Mins and maxs must be the same length.")
    # Get the dimension from one of the arrays
    dim = length(mins)
    # Initialize a Dataconfig with the explicit config
    DataConfig(
        true,   # setup
        mins,   # min
        maxs,   # max
        dim,    # dim
        dim*2   # dim_comp
    )
end # DataConfig(mins::RealVector, maxs::RealVector)

"""
    DataConfig(min::Real, max::Real, dim::Int)

Convenience constructor for DataConfig, requiring only a global min, max, and dim.

This constructor is used in the case that the feature mins and maxs are all the same respectively.
"""
function DataConfig(min::Real, max::Real, dim::Int)
    DataConfig(
        true,               # setup
        repeat([min], dim), # min
        repeat([max], dim), # max
        dim,                # dim
        dim*2               # dim_comp
    )
end # DataConfig(min::Real, max::Real, dim::Int)

"""
    element_min(x::RealVector, W::RealVector)

Returns the element-wise minimum between sample x and weight W.
"""
function element_min(x::RealVector, W::RealVector)
    # Compute the element-wise minimum of two vectors
    return minimum([x W], dims = 2)
end # element_min(x::RealVector, W::RealVector)

"""
    performance(y_hat::IntegerVector, y::IntegerVector)

Returns the categorization performance of y_hat against y.
"""
function performance(y_hat::IntegerVector, y::IntegerVector)
    # Get the number of labels
    n_y = length(y)

    # Check lengths
    if length(y_hat) != n_y
        error("Label vectors must be the same length")
    end

    # Get the number of correct classifications
    n_correct = 0
    for ix = 1:n_y
        if y_hat[ix] == y[ix]
            n_correct += 1
        end
    end

    return n_correct/n_y
end # performance(y_hat::IntegerVector, y::IntegerVector)

"""
    get_data_shape(data::RealArray)

Returns the correct feature dimension and number of samples.
"""
function get_data_shape(data::RealArray)
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
end # get_data_shape(data::RealArray)

"""
    get_n_samples(data::RealArray)

Returns the number of samples, accounting for 1-D and 2-D arrays.
"""
function get_n_samples(data::RealArray)
    # Get the correct dimensionality and number of samples
    if ndims(data) > 1
        n_samples = size(data)[2]
    else
        # n_samples = length(data)
        n_samples = 1
    end

    return n_samples
end # get_n_samples(data::RealArray)

"""
    data_setup!(config::DataConfig, data::RealMatrix)

Sets up the data config for the ART module before training.
"""
function data_setup!(config::DataConfig, data::RealMatrix)
    if config.setup
        @warn "Data configuration already set up, overwriting config"
    else
        config.setup = true
    end

    # Get the correct dimensionality and number of samples
    config.dim, _ = get_data_shape(data)
    config.dim_comp = 2*config.dim

    # Compute the ranges of each feature
    config.mins = [minimum(data[i, :]) for i in 1:config.dim]
    config.maxs = [maximum(data[i, :]) for i in 1:config.dim]
end # data_setup!(config::DataConfig, data::RealMatrix)

"""
    data_setup!(art::ART, data::RealMatrix)

Convenience method for setting up the DataConfig of an ART module in advance.
"""
function data_setup!(art::ART, data::RealMatrix)
    # Modify the DataConfig of the ART module directly
    data_setup!(art.config, data)
end # data_setup!(art::ART, data::RealMatrix)

"""
    DataConfig(data::RealMatrix)

Convenience constructor for DataConfig, requiring only the data matrix.
"""
function DataConfig(data::RealMatrix)
    # Create an empty dataconfig
    config = DataConfig()

    # Runthe setup upon the config using the data matrix for reference
    data_setup!(config, data)

    # Return the constructed DataConfig
    return config
end # DataConfig(min::Real, max::Real, dim::Int)

"""
    get_data_characteristics(data::RealArray ; config::DataConfig=DataConfig())

Get the characteristics of the data, taking account if a data config is passed.

If no DataConfig is passed, then the data characteristics come from the array itself. Otherwise, use the config for the statistics of the data and the data array for the number of samples.
"""
function get_data_characteristics(data::RealArray ; config::DataConfig=DataConfig())
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
end # get_data_characteristics(data::RealArray ; config::DataConfig=DataConfig())

"""
    linear_normalization(data::RealVector ; config::DataConfig=DataConfig())

Normalize the data to the range [0, 1] along each feature.
"""
function linear_normalization(data::RealVector ; config::DataConfig=DataConfig())
    # Vector normalization requires a setup DataConfig
    if !config.setup
        error("Attempting to complement code a vector without a setup DataConfig")
    end

    # Populate a new array with normalized values.
    x_raw = zeros(config.dim)

    # Iterate over each dimension
    for i = 1:config.dim
        denominator = config.maxs[i] - config.mins[i]
        if denominator != 0
            # If the denominator is not zero, normalize
            x_raw[i] = (data[i] .- config.mins[i]) ./ denominator
        else
            # Otherwise, the feature is zeroed because it contains no useful information
            x_raw[i] = zero(Int)
        end
    end
    return x_raw
end # linear_normalization(data::RealArray ; config::DataConfig=DataConfig())

"""
    linear_normalization(data::RealMatrix ; config::DataConfig=DataConfig())

Normalize the data to the range [0, 1] along each feature.
"""
function linear_normalization(data::RealMatrix ; config::DataConfig=DataConfig())
    # Get the data characteristics
    dim, n_samples, mins, maxs = get_data_characteristics(data, config=config)

    # Populate a new array with normalized values.
    x_raw = zeros(dim, n_samples)

    # Verify that all maxs are strictly greater than mins
    if !all(mins .< maxs)
        error("Got a data max index that is smaller than the corresonding min")
    end

    # Iterate over each dimension
    for i = 1:dim
        denominator = maxs[i] - mins[i]
        if denominator != 0
            # If the denominator is not zero, normalize
            x_raw[i, :] = (data[i, :] .- mins[i]) ./ denominator
        else
            # Otherwise, the feature is zeroed because it contains no useful information
            x_raw[i, :] = zeros(length(x_raw[i, :]))
        end
    end
    return x_raw
end # linear_normalization(data::RealMatrix ; config::DataConfig=DataConfig())

"""
    complement_code(data::RealArray ; config::DataConfig=DataConfig())

Normalize the data x to [0, 1] and returns the augmented vector [x, 1 - x].
"""
function complement_code(data::RealArray ; config::DataConfig=DataConfig())
    # Normalize the data
    x_raw = linear_normalization(data, config=config)

    # Complement code the data and return a concatenated matrix
    return vcat(x_raw, 1 .- x_raw)
end # complement_code(data::RealArray ; config::DataConfig=DataConfig())

"""
    get_iterator(opts::ARTOpts, x::Array)
"""
function get_iterator(opts::ARTOpts, x::RealArray)
    # Show a progbar only if the data is 2-D and the option is on
    dim, n_samples = get_data_shape(x)
    single_sample = n_samples == 1

    # Decide if using a progress bar or not
    # Don't use one if either there is a single sample or the display option is off
    prog_bar = single_sample ? false : opts.display

    # Construct the iterator
    iter_raw = 1:n_samples
    iter = prog_bar ? ProgressBar(iter_raw) : iter_raw

    return iter
end # get_iterator(opts::ARTOpts, x::RealArray)

"""
    update_iter(art::ARTModule, iter::ARTIterator, i::Int)
"""
function update_iter(art::ARTModule, iter::ARTIterator, i::Int)
    # Check explicitly for each, as the function definition restricts the types
    if iter isa ProgressBar
        set_description(iter, string(@sprintf("Ep: %i, ID: %i, Cat: %i", art.epoch, i, art.n_categories)))
    elseif iter isa UnitRange
        return
    end
end # update_iter(art::ARTModule, iter::Union{UnitRange, ProgressBar}, i::Int)

"""
    get_sample(x::RealArray, i::Int)

Returns a sample from data array x safely, accounting for 1-D and
"""
function get_sample(x::RealArray, i::Int)
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
end # get_sample(x::RealArray, i::Int)

"""
    init_train!(x::RealVector, art::ARTModule, preprocessed::Bool)
"""
function init_train!(x::RealVector, art::ARTModule, preprocessed::Bool)
    # If the data is not preprocessed
    if !preprocessed
        # If the data config is not setup, not enough information to preprocess
        if !art.config.setup
            error("$(typeof(art)): cannot preprocess data before being setup.")
        end
        x = complement_code(x, config=art.config)
    # If it is preprocessed and we are not setup
    elseif !art.config.setup
        # Get the dimension of the vector
        dim_comp = length(x)
        # If the complemented dimension is not even, error
        if !iseven(dim_comp)
            error("Declare that the vector is preprocessed, but it is not even")
        end
        # Half the complemented dimension and setup the DataConfig with that
        dim = Int(dim_comp/2)
        art.config = DataConfig(0, 1, dim)
    end
    return x
end # init_train!(x::RealVector, art::ARTModule, preprocessed::Bool)

"""
    init_train!(x::RealMatrix, art::ARTModule, preprocessed::Bool)
"""
function init_train!(x::RealMatrix, art::ARTModule, preprocessed::Bool)
    # If the data is not preprocessed, then complement code it
    if !preprocessed
        # Set up the data config if training for the first time
        !art.config.setup && data_setup!(art.config, x)
        x = complement_code(x, config=art.config)
    end
    return x
end # init_train!(x::RealMatrix, art::ART, preprocessed::Bool)

"""
    init_classify!(x::RealArray, art::ARTModule, preprocessed::Bool)
"""
function init_classify!(x::RealArray, art::ARTModule, preprocessed::Bool)
    # If the data is not preprocessed
    if !preprocessed
        # If the data config is not setup, not enough information to preprocess
        if !art.config.setup
            error("$(typeof(art)): cannot preprocess data before being setup.")
        end
        # Dispatch to the correct complement code method (vector or matrix)
        x = complement_code(x, config=art.config)
    end
    return x
end # init_classify!(x::RealArray, art::ART, preprocessed::Bool)


"""
    classify(art::ARTModule, x::RealMatrix ; preprocessed::Bool=false, get_bmu::Bool=false)

Predict categories of 'x' using the ART model.

Returns predicted categories 'y_hat.'

# Arguments
- `art::ARTModule`: ART or ARTMAP module to use for batch inference.
- `x::RealMatrix`: the 2-D dataset containing columns of samples with rows of features.
- `preprocessed::Bool=false`: flag, if the data has already been complement coded or not.
- `get_bmu::Bool=false`, flag, if the model should return the best-matching-unit label in the case of total mismatch.

# Examples
```julia-repl
julia> my_DDVFA = DDVFA()
DDVFA
    opts: opts_DDVFA
    ...
julia> x, y = load_data()
julia> train!(my_DDVFA, x)
julia> y_hat = classify(my_DDVFA, y)
```
"""
function classify(art::ARTModule, x::RealMatrix ; preprocessed::Bool=false, get_bmu::Bool=false)
    # Show a message if display is on
    art.opts.display && @info "Testing $(typeof(art))"

    # Preprocess the data
    x = init_classify!(x, art, preprocessed)

    # Data information and setup
    n_samples = get_n_samples(x)

    # Initialize the output vector
    y_hat = zeros(Int, n_samples)

    # Get the iterator based on the module options and data shape
    iter = get_iterator(art.opts, x)
    for ix = iter
        # Update the iterator if necessary
        update_iter(art, iter, ix)

        # Grab the sample slice
        sample = get_sample(x, ix)

        # Get the classification
        y_hat[ix] = classify(art, sample, preprocessed=true, get_bmu=get_bmu)
    end

    return y_hat
end # classify(art::ARTModule, x::RealMatrix ; preprocessed::Bool=false, get_bmu::Bool=false)

# -------------------------------------------
# Common Documentation
# -------------------------------------------

@doc raw"""
    classify(art::ARTModule, x::RealVector ; preprocessed::Bool=false, get_bmu::Bool=false)

Predict categories of a single sample of features 'x' using the ART model.

Returns predicted category 'y_hat.'

# Arguments
- `art::ART`: ART module to use for batch inference.
- `x::RealVector`: the single sample of features to classify.
- `preprocessed::Bool=false`: optional, flag if the data has already been complement coded or not.
- `get_bmu::Bool=false`: optional, flag if the model should return the best-matching-unit label in the case of total mismatch.
"""