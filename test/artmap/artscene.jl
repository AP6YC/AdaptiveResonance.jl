"""
    test_artscene.jl

# Description
A container for just ARTSCENE-specific unit tests.
"""

using
    Distributed,
    Logging

@testset "ARTSCENE Filter Porcelain" begin
    @info "------- ARTSCENE test -------"

    # Add four workers and give them all function definitions
    # addprocs(3)
    # @everywhere using AdaptiveResonance
    # Show the parallel workers
    # n_processes = nprocs()
    # n_workers = nworkers()
    # @info "Started parallel workers. Processes: $n_processes, Workers: $n_workers"

    # Set the logging level to Debug within the test
    ENV["JULIA_DEBUG"] = AdaptiveResonance

    # Random image
    raw_image = rand(3, 5, 5)

    # Process the image through the filters
    O, C = artscene_filter(raw_image)

    # Set the logging level back to Info
    ENV["JULIA_INFO"] = AdaptiveResonance

    # Close the workers after testing
    # rmprocs(workers())
    # n_processes = nprocs()
    # n_workers = nworkers()
    # @info "Closed parallel workers. Processes: $n_processes, Workers: $n_workers"
end
