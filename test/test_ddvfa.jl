function ddvfa_example()

    # Set the log level
    LogLevel(Logging.Info)

    # Parse the data
    data_file = "../data/art_data_rng.csv"
    train_x = readdlm(data_file, ',')
    train_x = permutedims(train_x)

    # Create the ART module, train, and classify
    art = DDVFA()
    train!(art, train_x)

    # Total number of categories
    total_vec = [art.F2[i].n_categories for i = 1:art.n_categories]
    total_cat = sum(total_vec)

    # # Calculate performance
    # perf = performance(y_hat, test_y)
    # println("Performance is ", perf)

    # # View the profile as a flamegraph
    # ProfileVega.view()
end
