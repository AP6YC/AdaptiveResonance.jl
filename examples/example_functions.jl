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

function sort_cvi_data(data::Array{N, 2}, labels::Array{M, 1}) where {N<:Real, M<:Int}
    index = sortperm(labels)
    data = data[:, index]
    labels = labels[index]

    return data, labels
end

# function sort_cvi_data!(data::Array{N, 2}, labels::Array{M, 1}) where {N<:Real, M<:Int}
#     data, labels = sort_cvi_data(data, labels)
# end