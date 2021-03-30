using DelimitedFiles

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
