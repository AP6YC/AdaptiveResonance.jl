using Random
using DelimitedFiles

"""
    DataSplit

A basic struct for encapsulating the four components of supervised training.
"""
struct DataSplit
    train_x::Array
    test_x::Array
    train_y::Array
    test_y::Array
    DataSplit(train_x, test_x, train_y, test_y) = new(train_x, test_x, train_y, test_y)
end # DataSplit

"""
    DataSplit(data_x::Array, data_y::Array, ratio::Float)

Return a DataSplit struct that is split by the ratio (e.g. 0.8).
"""
function DataSplit(data_x::Array, data_y::Array, ratio::Real)
    dim, n_data = size(data_x)
    split_ind = Int(floor(n_data*ratio))

    train_x = data_x[:, 1:split_ind]
    test_x = data_x[:, split_ind+1:end]
    train_y = data_y[1:split_ind]
    test_y = data_y[split_ind+1:end]

    return DataSplit(train_x, test_x, train_y, test_y)
end # DataSplit(data_x::Array, data_y::Array, ratio::Real)

"""
    tt_supervised(art::T, data::DataSplit) where {T<:AbstractART}

Train and test an ART module in a supervised manner on the dataset.
"""
function tt_supervised(art::T, data::DataSplit) where {T<:AbstractART}
    # Train and classify
    train!(art, data.train_x, data.train_y)
    y_hat = classify(art, data.test_x)

    # Calculate performance
    perf = performance(y_hat, data.test_y)
    @info "Performance is $perf"

    return perf
end # tt_supervised(art::T, data::DataSplit) where {T<:AbstractART}

"""
    showtypetree(T, level=0)

Show the tree of subtypes for a type.
```julia
showtypetree(Number)
```
"""
function showtypetree(T, level=0)
    println("\t" ^ level, T)
    for t in subtypes(T)
        showtypetree(t, level+1)
    end
end # showtypetree(T, level=0)

"""
    load_iris(data_path::String ; split_ratio::Real = 0.8)

Loads the iris dataset for testing and examples.
"""
function load_iris(data_path::String ; split_ratio::Real = 0.8)
    raw_data = readdlm(data_path,',')
    labels = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    raw_x = raw_data[2:end, 2:5]
    raw_y_labels = raw_data[2:end, 6]
    raw_y = Array{Int64}(undef, 0)
    for ix = 1:length(raw_y_labels)
        for jx = 1:length(labels)
            if raw_y_labels[ix] == labels[jx]
                push!(raw_y, jx)
            end
        end
    end
    n_samples, n_features = size(raw_x)

    # Julia is column-major, so use columns for features
    raw_x = permutedims(raw_x)

    # Shuffle the data and targets
    ind_shuffle = Random.randperm(n_samples)
    x = raw_x[:, ind_shuffle]
    y = raw_y[ind_shuffle]

    data = DataSplit(x, y, split_ratio)

    return data
end # load_iris(data_path::String ; split_ratio::Real = 0.8)

"""
    get_cvi_data(data_file::String)

Get the cvi data specified by the data_file path.
"""
function get_cvi_data(data_file::String)
    # Parse the data
    data = readdlm(data_file, ',')
    data = permutedims(data)
    train_x = data[1:2, :]
    train_y = convert(Array{Int64}, data[3, :])

    return train_x, train_y
end # get_cvi_data(data_file::String)

"""
    sort_cvi_data(data::Array{N, 2}, labels::Array{M, 1}) where {N<:Real, M<:Int}

Sorts the CVI data by the label index, assuring that clusters are provided incrementally.
"""
function sort_cvi_data(data::Array{N, 2}, labels::Array{M, 1}) where {N<:Real, M<:Int}
    index = sortperm(labels)
    data = data[:, index]
    labels = labels[index]

    return data, labels
end # sort_cvi_data(data::Array{N, 2}, labels::Array{M, 1}) where {N<:Real, M<:Int}

"""
    relabel_cvi_data(labels::Array{M, 1}) where {M<:Int}

Relabels the vector to present new labels in incremental order.
"""
function relabel_cvi_data(labels::Array{M, 1}) where {M<:Int}
    # Get the unique labels and their order of appearance
    unique_labels = unique(labels)
    n_unique_labels = length(unique_labels)
    n_labels = length(labels)

    # Create a new ordered list of unique labels
    new_unique_labels = [x for x in 1:n_unique_labels]

    # Map the old unique labels to the new ones
    label_mapping = Dict(zip(unique_labels, new_unique_labels))

    # Create a new labels vector with ordered labels
    new_labels = zeros(Int, n_labels)
    for ix = 1:n_labels
        new_labels[ix] = label_mapping[labels[ix]]
    end

    return new_labels
end # relabel_cvi_data(labels::Array{M, 1}) where {M<:Int}

"""
    get_bernoulli_subset(data::Array{N, 2}, labels::Array{M, 1}) where {N<:Real, M<:Int}
"""
function get_bernoulli_subset(data::Array{N, 2}, labels::Array{M, 1}, p::Float64) where {N<:Real, M<:Int}
    # Get the dimensions of the data
    dim, n_samples = size(data)

    # Get a random subsamplin of the data
    subset = randsubseq(1:n_samples, p)

    # Return the subset
    return data[:, subset], labels[subset]
end