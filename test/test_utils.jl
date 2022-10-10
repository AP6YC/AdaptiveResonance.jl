"""
    test_utils.jl

A set of common struct and function utilities for AdaptiveResonance.jl unit tests.
"""

# --------------------------------------------------------------------------- #
# IMPORTS
# --------------------------------------------------------------------------- #

using DelimitedFiles
using NumericalTypeAliases

# --------------------------------------------------------------------------- #
# STRUCTS
# --------------------------------------------------------------------------- #

"""
A basic struct for encapsulating the four components of supervised training.
"""
struct DataSplit
    """
    The training feature samples.
    Dimensions are `(feature-dim, sample-index)`.
    """
    train_x::Matrix{Float}

    """
    The testing feature samples.
    Dimensions are `(feature-dim, sample-index)`.
    """
    test_x::Matrix{Float}

    """
    A vector of training labels.
    """
    train_y::Vector{Int}

    """
    A vector of testing labels.
    """
    test_y::Vector{Int}

    DataSplit(train_x, test_x, train_y, test_y) = new(train_x, test_x, train_y, test_y)
end # DataSplit

# --------------------------------------------------------------------------- #
# METHODS
# --------------------------------------------------------------------------- #

"""
Return a DataSplit struct that is split by the ratio (e.g. 0.8).

# Arguments
- `data_x::RealMatrix`: a 2-D matrix of samples with convention (features, samples).
- `data_y::RealVector`: a 1-D vector of integered labels.
- `ratio::Real`: the ratio for the train/test split ∈ (0, 1).
"""
function DataSplit(data_x::RealMatrix, data_y::RealVector, ratio::Real)
    _, n_data = size(data_x)
    split_ind = Integer(floor(n_data*ratio))

    train_x = data_x[:, 1:split_ind]
    test_x = data_x[:, split_ind + 1:end]
    train_y = data_y[1:split_ind]
    test_y = data_y[split_ind + 1:end]

    return DataSplit(train_x, test_x, train_y, test_y)
end # DataSplit(data_x::RealMatrix, data_y::RealVector, ratio::Real)

# --------------------------------------------------------------------------- #
# FUNCTIONS
# --------------------------------------------------------------------------- #

"""
Train and test an ART module.

# Arguments
- `art::ARTModule`: the ART or ARTMAP module to train and test.
- `data::DataSplit`: the struct containing a train/test split.
- `supervised::Bool=false`: flag for using supervised learning for ART modules (i.e., ARTMAP modules are always supervised).
- `train_opts::NamedTuple=NamedTuple()`: keyword options to pass to the `train!` function.
- `test_opts::NamedTuple=NamedTuple()`: keyword options to pass to the `classify` function.
"""
function train_test_art(
    art::ARTModule,
    data::DataSplit ;
    supervised::Bool=false,
    train_opts::NamedTuple=NamedTuple(),
    test_opts::NamedTuple=NamedTuple()
)
    # Default performance to undefined
    perf = NaN
    # If the module is unsupervised by default
    if art isa ART
        # Check if training with lazy supervision
        if supervised
            # Train with the supervised ART syntax
            train!(art, data.train_x, y=data.train_y; train_opts...)
            y_hat = classify(art, data.test_x; test_opts...)

            # Calculate performance
            perf = performance(y_hat, data.test_y)
        # Otherwise, train in an unsupervised fashion
        else
            train!(art, data.train_x; train_opts...)
        end
    # Otherwise, necessarily train on a supervised model
    elseif art isa ARTMAP
        # Train and classify
        train!(art, data.train_x, data.train_y; train_opts...)
        y_hat = classify(art, data.test_x; test_opts...)

        # Calculate performance
        perf = performance(y_hat, data.test_y)
    else
        error("Incompatible ART module passed for testing")
    end

    # If the performance is not a NaN (potentially unsupervsied), then display perf
    if !isnan(perf)
        @info "$(typeof(art)): performance is $perf"
    end

    return perf
end

"""
Loads the iris dataset for testing and examples.

# Arguments
- `data_path::AbstractString`: path containing the Iris dataset.
- `split_ratio::Real = 0.8`: train/test split ration ∈ (0, 1).
"""
function load_iris(data_path::AbstractString ; split_ratio::Real = 0.8)
    raw_data = readdlm(data_path, ',')
    labels = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    raw_x = Matrix{Real}(raw_data[2:end, 2:5])
    raw_y_labels = raw_data[2:end, 6]
    raw_y = Vector{Int}(undef, 0)
    for ix in eachindex(raw_y_labels)
        for jx in eachindex(labels)
            if raw_y_labels[ix] == labels[jx]
                push!(raw_y, jx)
            end
        end
    end

    # Julia is column-major, so use columns for features
    raw_x = permutedims(raw_x)

    # Create the datasplit object
    data = DataSplit(raw_x, raw_y, split_ratio)

    return data
end # load_iris(data_path::AbstractString ; split_ratio::Real = 0.8)
