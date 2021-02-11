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

# """
#     load_am_data(N_train, N_test)

# Loads the ARTMAP test data, cutting off at N_train training data points and
# N_test testing data points. In this case, it loads the MNIST handwritten digits
# dataset and packages them into a DataSplit struct.
# """
# function load_am_data(N_train::Int, N_test::Int)
#     # Load the data, downloading if in a CI context: TODO
#     # if ENV["CI"] == true
#     data_dir = "../data/mnist/"
#     if !isdir(data_dir)
#         MNIST.download(data_dir, i_accept_the_terms_of_use=true)
#     end
#     # end
#     train_x, train_y = MNIST.traindata(dir=data_dir)
#     test_x, test_y = MNIST.testdata(dir=data_dir)

#     # Get sizes of train and test data
#     size_a, size_b, data_n = size(train_x)
#     size_at, size_bt, data_nt = size(test_x)

#     # Take the minimum of the user N and the number of data
#     N_train = minimum([N_train, data_n])
#     N_test = minimum([N_test, data_nt])

#     # Permute these images because the MLDatasets package is goofy
#     for i=1:N_train
#         train_x[:,:,i] = permutedims(train_x[:,:,i])
#     end
#     for i=1:N_test
#         test_x[:,:,i] = permutedims(test_x[:,:,i])
#     end

#     # Flatten the arrays
#     train_x_flat = zeros(size_a*size_b, N_train)
#     test_x_flat = zeros(size_at*size_bt, N_test)
#     for i=1:N_train
#         train_x_flat[:, i] = train_x[:,:, i][:]
#     end
#     for i=1:N_test
#         test_x_flat[:, i] = test_x[:,:, i][:]
#     end
#     train_y = train_y[1:N_train]
#     test_y = test_y[1:N_test]

#     # Create the data struct in a nice package to pass to testing functions
#     data = DataSplit(train_x_flat, test_x_flat, train_y, test_y)

#     return data
# end
