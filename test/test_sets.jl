using AdaptiveResonance
using Test
using MLDatasets
using Logging
using DelimitedFiles
using Random

Random.seed!(0)

# Auxiliary generic functions for loading data, etc.
include("test_utils.jl")

@testset "ARTSCENE.jl" begin

    # ARTSCENE training and testing
    include("test_artscene.jl")

end

@testset "DDVFA.jl" begin

    # DDVFA train and test functions
    include("test_ddvfa.jl")

end

@testset "AdaptiveResonance.jl" begin

    include("modules.jl")

end

@testset "ARTMAP.jl" begin

    # ARTMAP training and testing functions
    include("test_sfam.jl")
    # data = load_am_data(200, 50)
    data = load_iris("../data/Iris.csv")
    sfam_example(data)
    dam_example(data)

    # Iris training
    # data = load_iris("../data/Iris.csv")
    # # Create the ART module, train, and classify
    # art = SFAM()
    # train!(art, data.train_x, data.train_y)
    # y_hat = classify(art, data.test_x)

    # # Calculate performance
    # @info size(y_hat)
    # @info size(data.test_y)
    # perf = performance(y_hat, data.test_y)
    # println("Performance is ", perf)
end
