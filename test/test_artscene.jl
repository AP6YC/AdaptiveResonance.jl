"""
    artscene_example()

Trains and tests a Default ARTMAP module with input data.
"""
function artscene_example()
    # Set the logging level to Info
    LogLevel(Logging.Info)

    # include("../src/ARTMAP/ARTSCENE.jl")
    # Random image
    image = rand(50, 50, 3)

    # Stage 1: Grayscale
    image = color_to_gray(image)
    println("Before contrast")

    # Stage 2: Contrast normalization
    x = contrast_normalization(image)
    println("After contrast")

    # Stage 3:
    # Create the ART module, train, and classify
    # art = DAM()
    # train!(art, data.train_x, data.train_y)
    # y_hat = classify(art, data.test_x)

    # # Calculate performance
    # perf = performance(y_hat, data.test_y)
    # println("Performance is ", perf)
end