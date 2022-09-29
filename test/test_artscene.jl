"""
    test_artscene.jl

A container for just ARTSCENE-specific unit tests.
"""

using Distributed
using Logging

"""
    artscene_filter_porcelain()

Runs the artscene user-level functions on a random image.
"""
# function artscene_filter_porcelain()
@testset "ARTSCENE Filter Porcelain" begin
    @info "------- ARTSCENE test -------"

    # Add four workers and give them all function definitions
    addprocs(4)
    @everywhere using AdaptiveResonance

    # Show the parallel workers
    n_processes = nprocs()
    n_workers = nworkers()
    @info "Started parallel workers. Processes: $n_processes, Workers: $n_workers"

    # Set the logging level to Debug within the test
    LogLevel(Logging.Debug)

    # Random image
    raw_image = rand(3, 5, 5)

    # Process the image through the filters
    O, C = artscene_filter(raw_image)

    # Set the logging level back to Info
    LogLevel(Logging.Info)

    # Close the workers after testing
    rmprocs(workers())
    n_processes = nprocs()
    n_workers = nworkers()
    @info "Closed parallel workers. Processes: $n_processes, Workers: $n_workers"
end # @testset "ARTSCENE Filter Porcelain"
