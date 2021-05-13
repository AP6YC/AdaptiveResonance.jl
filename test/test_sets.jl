using AdaptiveResonance
using Test
using Logging
using DelimitedFiles
using Random

# Set the log level
LogLevel(Logging.Info)

# Auxiliary generic functions for loading data, etc.
include("test_utils.jl")

@testset "common.jl" begin
    # Set the logging level to Info and standardize the random seed
    LogLevel(Logging.Info)
    Random.seed!(0)

    # Load the data and test across all supervised modules
    data = load_iris("../data/Iris.csv")

    # Example arrays
    three_by_two = [1 2; 3 4; 5 6]

    # Test DataConfig constructors
    dc1 = DataConfig()                  # Default constructor
    dc2 = DataConfig(0, 1, 2)           # When min and max are same across all features
    dc3 = DataConfig([0, 1], [2, 3])    # When min and max differ across features

    # Test get_n_samples
    @test get_n_samples([1,2,3]) == 1           # 1-D array case
    @test get_n_samples(three_by_two) == 2      # 2-D array case

    # Test breaking situations
    @test_throws ErrorException performance([1,2],[1,2,3])
    @test_logs (:warn,) AdaptiveResonance.data_setup!(dc3, three_by_two)
    bad_config =  DataConfig(1, 0, 3)
    @test_throws ErrorException linear_normalization(three_by_two, config=bad_config)

end

@testset "constants.jl" begin
    @info "Constants testing"
    ddvfa_methods = ["single",
                     "average",
                     "complete",
                     "median",
                     "weighted",
                     "centroid"]
    @test AdaptiveResonance.DDVFA_METHODS == ddvfa_methods
end

@testset "DDVFA.jl" begin
    # DDVFA training and testing
    include("test_ddvfa.jl")
end # @testset "DDVFA.jl"

@testset "AdaptiveResonance.jl" begin
    # Module loading
    include("modules.jl")
end # @testset "AdaptiveResonance.jl"

@testset "ARTMAP.jl" begin
    # Set the logging level to Info and standardize the random seed
    LogLevel(Logging.Info)
    Random.seed!(0)

    # Load the data and test across all supervised modules
    data = load_iris("../data/Iris.csv")
    for art in [SFAM, DAM]
        perf = tt_supervised(art(), data)
        @test perf > 0.8
    end
end # @testset "ARTMAP.jl"

@testset "ARTSCENE.jl" begin
    # ARTSCENE training and testing
    include("test_artscene.jl")
end # @testset "ARTSCENE.jl"
