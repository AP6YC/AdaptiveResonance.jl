"""
    common.jl

# Description
Common algorithmic types and functions used throughout the package.
"""

# -----------------------------------------------------------------------------
# TYPE ALIASES
# -----------------------------------------------------------------------------

"""
Definition of the ART module statistics dictionary, used to generate and store various logs during training and testing.
"""
const ARTStats = Dict{String, Any}

# -----------------------------------------------------------------------------
# TYPES
# -----------------------------------------------------------------------------

"""
Container to standardize training/testing data configuration.

This container declares if a data configuration has been setup, what the original and complement coded dimensions are, and what the minimums and maximums of the values along each feature dimension are.
"""
mutable struct DataConfig
    """
    Flag if data has been setup yet or not.
    """
    setup::Bool

    """
    List of minimum values for each feature.
    """
    mins::Vector{Float}

    """
    List of maximum values for each feature.
    """
    maxs::Vector{Float}

    """
    Dimensionality of the feature vectors (i.e., number of features).
    """
    dim::Int

    """
    Complement coded feature dimensionality, twice the size of `dim`.
    """
    dim_comp::Int
end

# -----------------------------------------------------------------------------
# CONSTRUCTORS
# -----------------------------------------------------------------------------

"""
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
end

"""
Convenience constructor for DataConfig, requiring only mins and maxs of the features.

This constructor is used when the mins and maxs differ across features. The dimension is inferred by the length of the mins and maxs.

# Arguments
- `mins::RealVector`: a vector of minimum values for each feature dimension.
- `maxs::RealVector`: a vector of maximum values for each feature dimension.
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
        dim * 2 # dim_comp
    )
end

"""
Convenience constructor for DataConfig, requiring only a global min, max, and dim.

This constructor is used in the case that the feature mins and maxs are all the same respectively.

# Arguments
- `min::Real`: the minimum value across all features.
- `max::Real`: the maximum value across all features.
- `dim::Integer`: the dimension of the features, which must be provided because it cannot be inferred from just the minimum or maximum values.
"""
function DataConfig(min::Real, max::Real, dim::Integer)
    DataConfig(
        true,               # setup
        repeat([min], dim), # min
        repeat([max], dim), # max
        dim,                # dim
        dim * 2             # dim_comp
    )
end

"""
Convenience constructor for DataConfig, requiring only the data matrix.

# Arguments
- `data::RealMatrix`: the 2-D batch of data to be used for inferring the data configuration.
"""
function DataConfig(data::RealMatrix)
    # Create an empty dataconfig
    config = DataConfig()

    # Runthe setup upon the config using the data matrix for reference
    data_setup!(config, data)

    # Return the constructed DataConfig
    return config
end

# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------

"""
Initializes an ARTStats dictionary with zero entries.
"""
function build_art_stats()
    # Create the stats dictionary
    stats = ARTStats()

    # Initialize zero entries for each element
    stats["M"] = 0.0
    stats["T"] = 0.0
    stats["bmu"] = 0
    stats["mismatch"] = false

    # Return the zero-initialized stats dictionary
    return stats
end

"""
Logs common statistics of an ART module after a training/classification iteration.

# Arguments
- `art::ARTModule`: the ART module that just underwent training/classification.
- `bmu::Integer`: the best-matching unit integer index.
- `mismatch::Bool`: flag of whether there was a mismatch in this iteration.
"""
function log_art_stats!(art::ARTModule, bmu::Integer, mismatch::Bool)
    # Overwrite the stats entries
    art.stats["M"] = art.M[bmu]
    art.stats["T"] = art.T[bmu]
    art.stats["bmu"] = bmu
    art.stats["mismatch"] = mismatch

    # Return empty
    return
end

"""
Returns the element-wise minimum between sample x and weight W.

# Arguments
- `x::RealVector`: the input sample.
- `W::RealVector`: the weight vector to compare the sample against.
"""
function element_min(x::RealVector, W::RealVector)
    # Get the length of the sample
    n_el = length(x)
    # Create a destination in memory of zeros of type and size like the sample
    min_vec = zero(x)
    # Iterate over every element of the sample
    for ix = 1:n_el
        # Get and assign the minimum of the sample and weight at index ix
        @inbounds min_vec[ix] = min(x[ix], W[ix])
    end
    # Return the element-minimum vector
    return min_vec
    # return @inbounds vec(minimum([x W], dims = 2))
end

"""
Convenience function to get the categorization performance of y_hat against y.

# Arguments
- `y_hat::IntegerVector`: the estimated labels.
- `y::IntegerVector`: the true labels.
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

    # Return the performance as the number correct over the total number
    return n_correct/n_y
end

"""
Returns the dimension of the data, enforcint the (dim, n_samples) convention of the package.

# Arguments
- `data::RealMatrix`: the 2-D data to infer the feature dimension of.
"""
function get_dim(data::RealMatrix)
    # Return the correct dimension of the data
    return size(data)[ART_DIM]
end

"""
Returns the number of samples, enforcing the convention of the package.

# Arguments
- `data::RealMatrix`: the 2-D data to infer the number of samples from.
"""
function get_n_samples(data::RealMatrix)
    # Return the correct number of samples
    return size(data)[ART_SAMPLES]
end

"""
Returns the (dim, n_samples) of the provided 2-D data matrix, enforcing the ART package convention.

# Arguments
- `data::RealMatrix`: the 2-D data to infer the feature dimension and number of samples from.
"""
function get_data_shape(data::RealMatrix)
    # Get the dimension of the data
    dim = get_dim(data)
    # Get the number of samples of the data
    n_samples = get_n_samples(data)

    # Return the dimension and number of samples
    return dim, n_samples
end

"""
Sets up the data config for the ART module before training.

This function crucially gets the original and complement-coded dimensions of the data, and it infers the bounds of the data (minimums and maximums) by the largest and smallest values along each feature dimension.

# Arguments
- `config::DataConfig`: the ART/ARTMAP module's data configuration object.
- `data::RealMatrix`: the 2-D batch of data to use for creating the data configuration.
"""
function data_setup!(config::DataConfig, data::RealMatrix)
    if config.setup
        @warn "Data configuration already set up, overwriting config"
    else
        config.setup = true
    end

    # Get the correct dimensionality and number of samples
    config.dim = get_dim(data)
    config.dim_comp = 2 * config.dim

    # Compute the ranges of each feature
    config.mins = zeros(config.dim)
    config.maxs = zeros(config.dim)
    for ix = 1:config.dim
        config.mins[ix] = minimum(data[ix, :])
        config.maxs[ix] = maximum(data[ix, :])
    end
end

"""
Convenience method for setting up the DataConfig of an ART module in advance.

# Arguments
- `art::ARTModule`: the ART/ARTMAP module to manually configure the data config for.
- `data::RealArray`: the 2-D batch of data used to create the data config.
"""
function data_setup!(art::ARTModule, data::RealMatrix)
    # Modify the DataConfig of the ART module directly
    data_setup!(art.config, data)
end

"""
Get the characteristics of the data, taking account if a data config is passed.

If no DataConfig is passed, then the data characteristics come from the array itself.
Otherwise, use the config for the statistics of the data and the data array for the number of samples.

# Arguments
- `data::RealMatrix`: the 2-D data to be complement coded.
- `config::DataConfig=DataConfig()`: the data configuration for the ART/ARTMAP module.
"""
function get_data_characteristics(data::RealMatrix ; config::DataConfig=DataConfig())
    # If the data is setup, use the config
    if config.setup
        # Just get the number of samples and use the config for everything else
        n_samples = get_n_samples(data)
        dim = config.dim
        mins = config.mins
        maxs = config.maxs
    else
        # Get the feature dimension and number of samples
        dim, n_samples = get_data_shape(data)
        # Get the ranges for each feature
        mins = zeros(dim)
        maxs = zeros(dim)
        for ix = 1:dim
            mins[ix] = minimum(data[ix, :])
            maxs[ix] = maximum(data[ix, :])
        end
    end
    return dim, n_samples, mins, maxs
end

"""
Normalize the data to the range [0, 1] along each feature.

# Arguments
- `data::RealVector`: the 1-D sample of data to normalize.
- `config::DataConfig=DataConfig()`: the data configuration from the ART/ARTMAP module.
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
end

"""
Normalize the data to the range [0, 1] along each feature.

# Arguments
- `data::RealMatrix`: the 2-D batch of data to normalize.
- `config::DataConfig=DataConfig()`: the data configuration from the ART/ARTMAP module.
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
end

"""
Normalizes the data x to [0, 1] and returns the augmented vector [x, 1 - x].

# Arguments
- `data::RealArray`: the 1-D or 2-D data to be complement coded.
- `config::DataConfig=DataConfig()`: the data configuration for the ART/ARTMAP module.
"""
function complement_code(data::RealArray ; config::DataConfig=DataConfig())
    # Normalize the data
    x_raw = linear_normalization(data, config=config)

    # Complement code the data and return a concatenated matrix
    return vcat(x_raw, 1 .- x_raw)
end

"""
Creates an iterator object according to the ART/ARTMAP modules display settings for batch iteration.

# Arguments
- `opts::ARTOpts`: the ART/ARTMAP module's options containing display settings.
- `n_samples::Integer`: the number of iterations to create the iterator for.
"""
function get_iterator(opts::ARTOpts, n_samples::Integer)
    # Construct the iterator
    iter_raw = 1:n_samples

    # If we want a progress bar, construct one. Otherwise, return the raw iterator
    iter = opts.display ? ProgressBar(iter_raw) : iter_raw

    # Return the selected iterator
    return iter
end

"""
Updates the iteration of the ART/ARTMAP module, training or inference, according to its display settings.

# Arguments
- `art::ARTModule`: the ART/ARTMAP module being iterated upon.
- `iter::ARTIterator`: the iterator object used in the training/inference loop.
- `i::Integer`: the iteration during training/inference that the iterator should be updated to.
"""
function update_iter(art::ARTModule, iter::ARTIterator, i::Integer)
    # Check explicitly for each, as the function definition restricts the types
    if iter isa ProgressBar
        set_description(iter, "Ep: $(art.epoch), ID: $(i), Cat: $(art.n_categories)")
    elseif iter isa UnitRange
        return
    end
end

"""
Returns a sample from data array `x` at sample location `i`.
This function implements the convention that columns are samples while rows are features within samples.

# Arguments
- `x::RealMatrix`: the batch of data to grab a sample from.
- `i::Integer`: the index to get the sample from.
"""
function get_sample(x::RealMatrix, i::Integer)
    # Return the sample at location
    # return x[:, i]
    return @inbounds x[:, i]
end

"""
Initializes the module for training in a single iteration.

The purpose of this function is mainly to handle the conditions of complement coding.
Fails if the module was incorrectly set up or if the module was not setup and the data was not preprocessed.

# Arguments
- `x::RealVector`: the sample used for initialization.
- `art::ARTModule`: the ART/ARTMAP module that will be trained on the sample.
- `preprocessed::Bool`: a required flag for if the sample has already been complement coded and normalized.
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
            error("Declared that the vector is preprocessed, but it is not even")
        end
        # Half the complemented dimension and setup the DataConfig with that
        dim = Int(dim_comp/2)
        art.config = DataConfig(0, 1, dim)
    end
    return x
end

"""
Initializes the training loop for batch learning.

# Arguments
- `x::RealMatrix`: the data that is used for training.
- `art::ARTModule`: the ART/ARTMAP that will be trained.
- `preprocessed::Bool`: required flag for if the data has already been complement coded and normalized.
"""
function init_train!(x::RealMatrix, art::ARTModule, preprocessed::Bool)
    # If the data is not preprocessed, then complement code it
    if !preprocessed
        # Set up the data config if training for the first time
        !art.config.setup && data_setup!(art.config, x)
        x = complement_code(x, config=art.config)
    end
    return x
end

"""
Initializes the classification loop for batch inference.

# Arguments
- `x::RealArray`: the data that is used for inference.
- `art::ARTModule`: the ART/ARTMAP module that will be used for inference.
- `preprocessed::Bool`: required flag for if the data has already been complement coded and normalized.
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
end

"""
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
    iter = get_iterator(art.opts, n_samples)
    for ix = iter
        # Update the iterator if necessary
        update_iter(art, iter, ix)

        # Grab the sample slice
        sample = get_sample(x, ix)

        # Get the classification
        y_hat[ix] = classify(art, sample, preprocessed=true, get_bmu=get_bmu)
    end

    return y_hat
end

"""
Checks the stopping conditions for an ART module.

# Arguments
- `art::ART`: the ART module to check stopping conditions for.
"""
function stopping_conditions(art::ARTModule)
    return art.epoch >= art.opts.max_epoch
end
