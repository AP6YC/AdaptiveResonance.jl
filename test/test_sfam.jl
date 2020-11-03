"""
    sfam_example(data)

Trains and tests a Simple Fuzzy ARTMAP module with input data.
"""
function sfam_example(data)
    # Set the logging level to Info
    LogLevel(Logging.Info)

    # Create the ART module, train, and classify
    art = SFAM()
    train!(art, data.train_x, data.train_y)
    y_hat = classify(art, data.test_x)

    # Calculate performance
    perf = performance(y_hat, data.test_y)
    println("Performance is ", perf)
end


"""
    dam_example(data)

Trains and tests a Default ARTMAP module with input data.
"""
function dam_example(data)
    # Set the logging level to Info
    LogLevel(Logging.Info)

    # Create the ART module, train, and classify
    art = DAM()
    train!(art, data.train_x, data.train_y)
    y_hat = classify(art, data.test_x)

    # Calculate performance
    perf = performance(y_hat, data.test_y)
    println("Performance is ", perf)
end