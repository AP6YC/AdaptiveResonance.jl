function sfam_example()
    # Set the logging level to Info
    LogLevel(Logging.Info)

    # Load the data, downloading if in a CI context: TODO
    # if ENV["CI"] == true
    MNIST.download("../data/mnist/", i_accept_the_terms_of_use=true)
    # end
    train_x, train_y = MNIST.traindata()
    test_x, test_y = MNIST.testdata()

    # Get sizes of train and test data
    size_a, size_b, data_n = size(train_x)
    size_at, size_bt, data_nt = size(test_x)

    # Cut off the number of data points and flatten
    N_train = 200
    N_test = 50

    # Take the minimum of the user N and the number of data
    N_train = minimum([N_train, data_n])
    N_test = minimum([N_test, data_nt])

    # Permute these images because the MLDatasets package is goofy
    for i=1:N_train
        train_x[:,:,i] = permutedims(train_x[:,:,i])
    end
    for i=1:N_test
        test_x[:,:,i] = permutedims(test_x[:,:,i])
    end

    # Flatten the arrays
    train_x_flat = zeros(size_a*size_b, N_train)
    test_x_flat = zeros(size_at*size_bt, N_train)
    for i=1:N_train
        train_x_flat[:, i] = train_x[:,:, i][:]
    end
    for i=1:N_test
        test_x_flat[:, i] = test_x[:,:, i][:]
    end
    train_y = train_y[1:N_train]
    test_y = test_y[1:N_test]

    # Create the ART module, train, and classify
    art = SFAM()
    train!(art, train_x_flat, train_y)
    y_hat = classify(art, test_x_flat)

    # Calculate performance
    # perf = performance(y_hat, test_y)
    # println("Performance is ", perf)
end
