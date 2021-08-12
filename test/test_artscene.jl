using Distributed
using Logging

# Add four workers and give them all function definitions
addprocs(4)
@everywhere using AdaptiveResonance

"""
    artscene_filter_porcelain()

Runs the artscene user-level functions on a random image.
"""
# function artscene_filter_porcelain()
@testset "ARTSCENE Filter Porcelain" begin
    # Set the logging level to Debug
    LogLevel(Logging.Debug)
    n_processes = nprocs()
    n_workers = nworkers()
    @info "Processes: $n_processes, Workers: $n_workers"

    # Random image
    raw_image = rand(3, 5, 5)

    # Process the image through the filters
    O, C = artscene_filter(raw_image)

    # Set the logging level back to Info
    LogLevel(Logging.Info)
end # @testset "ARTSCENE Filter Porcelain"

# Close the workers after testing
rmprocs(workers())
