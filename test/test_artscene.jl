using Distributed
using Logging
addprocs(4)
@everywhere using AdaptiveResonance

"""
    artscene_filter_plumbing()

Runs a random image through each artscene filter function with logging.
"""
# function artscene_filter_plumbing()
@testset "ARTSCENE Filter Plumbing" begin
    # Set the logging level to Info
    LogLevel(Logging.Info)
    n_processes = nprocs()
    n_workers = nworkers()
    @info "Processes: $n_processes, Workers: n_workers"

    # Random image
    # raw_image = rand(3, 6, 6)
    raw_image = rand(3, 5, 5)
    image_size = size(raw_image)
    image_type =  typeof(raw_image)
    @info "Original: Size = $image_size, Type = $image_type"

    # Stage 1: Grayscale
    image = color_to_gray(raw_image)
    image_size = size(image)
    image_type = typeof(image)
    @info "Stage 1: Grayscale: Size = $image_size, Type = $image_type"
    @info "Stage 1: Done"

    # Stage 2: Contrast normalization
    x = contrast_normalization(image, distributed=true)
    image_size = size(x)
    image_type = typeof(x)
    @info "Stage 2: Contrast: Size = $image_size, Type = $image_type"
    @info "Stage 2: Done"

    # Stage 3: Contrast-sensitive oriented filtering
    y = contrast_sensitive_oriented_filtering(image, x)
    image_size = size(y)
    image_type = typeof(y)
    @info "Stage 3: Sensitive Oriented: Size = $image_size, Type = $image_type"
    @info "Stage 3: Done"

    # Stage 4: Contrast-insensitive oriented filtering
    z = contrast_insensitive_oriented_filtering(y)
    image_size = size(z)
    image_type = typeof(z)
    @info "Stage 4: Insensitive Oriented: Size = $image_size, Type = $image_type"
    @info "Stage 4: Done"

    # Stage 5: Orientation competition
    z = orientation_competition(z)
    image_size = size(z)
    image_type = typeof(z)
    @info "Stage 5: Orientation Competition: Size = $image_size, Type = $image_type"
    @info "Stage 5: Done"

    # *Stage 6*: Compute patch vectors (orientation and color)
    # O, C = patch_orientation_color(z, matrix_raw_image)
    O, C = patch_orientation_color(z, raw_image)
    @info "Stage 6: Done"

    # Stage 3:
    # Create the ART module, train, and classify
    # art = DAM()
    # train!(art, data.train_x, data.train_y)
    # y_hat = classify(art, data.test_x)

    # # Calculate performance
    # perf = performance(y_hat, data.test_y)
    # println("Performance is ", perf)
end

"""
    artscene_filter_porcelain()

Runs the artscene user-level functions on a random image.
"""
# function artscene_filter_porcelain()
@testset "ARTSCENE Filter Porcelain" begin
    # Set the logging level to Info
    LogLevel(Logging.Info)
    n_processes = nprocs()
    n_workers = nworkers()

    # Random image
    # raw_image = rand(3, 6, 6)
    raw_image = rand(3, 5, 5)

    @info "Processes: $n_processes, Workers: n_workers"
    O, C = artscene_filter(raw_image)
end