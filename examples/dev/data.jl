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
