using Distributed
using Logging
addprocs(4)
@everywhere using AdaptiveResonance

"""
    artscene_example()

Trains and tests a Default ARTMAP module with input data.
"""
function artscene_example()
    # Set the logging level to Info
    LogLevel(Logging.Info)
    # println("Processes: ", nprocs())
    @info "Processes", nprocs()
    # println("Workers: ", nworkers())
    @info "Workers", nworkers()

    # Random image
    image = rand(3, 8, 8)
    # println("Original: Size: ", size(image), "Type: ", typeof(image))
    image_size = size(image)
    image_type =  typeof(image)
    @info "Original: Size = $image_size, Type = $image_type"

    # Stage 1: Grayscale
    image = color_to_gray(image)
    # println("Before contrast: Size: ", size(image), ", Type: ", typeof(image))
    image_size = size(image)
    image_type = typeof(image)
    @info "Grayscale: Size = $image_size, Type = $image_type"
    @info "Stage 1 Done"

    # Stage 2: Contrast normalization
    x = contrast_normalization(image, distributed=true)
    image_size = size(x)
    image_type = typeof(x)
    @info "Contrast: Size = $image_size, Type = $image_type"
    @info "Stage 2 Done"

    # Stage 3: Contrast-sensitive oriented filtering
    y = contrast_sensitive_oriented_filtering(image, x)
    image_size = size(y)
    image_type = typeof(y)
    @info "Sensitive Oriented: Size = $image_size, Type = $image_type"
    @info "Stage 3 Done"

    # Stage 4: Contrast-insensitive oriented filtering
    z = contrast_insensitive_oriented_filtering(y)
    image_size = size(z)
    image_type = typeof(z)
    @info "Insensitive Oriented: Size = $image_size, Type = $image_type"
    @info "Stage 4 Done"

    # Stage 5: Orientation competition
    z = orientation_competition(z)
    image_size = size(z)
    image_type = typeof(z)
    @info "Orientation Competition: Size = $image_size, Type = $image_type"
    @info "Stage 5 Done"

    # *Stage 6*: Compute patch vectors (orientation and color)
    # O, C = patch_orientation_color(z, matrix_raw_image)

    # Stage 3:
    # Create the ART module, train, and classify
    # art = DAM()
    # train!(art, data.train_x, data.train_y)
    # y_hat = classify(art, data.test_x)

    # # Calculate performance
    # perf = performance(y_hat, data.test_y)
    # println("Performance is ", perf)
end