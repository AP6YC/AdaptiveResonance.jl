using AdaptiveResonance
using Test
using Logging
using DelimitedFiles
using Random

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
    # Set the logging level to Info
    LogLevel(Logging.Info)
    Random.seed!(0)

    # ARTMAP training and testing functions
    # data = load_am_data(200, 50)
    data = load_iris("../data/Iris.csv")

    for art in [SFAM, DAM]
        # sfam = SFAM()
        perf = tt_supervised(art(), data)
        @test perf > 0.9
    end
    # dam = DAM()
    # perf = tt_supervised(dam, data)
    # @test perf > 0.9

end
