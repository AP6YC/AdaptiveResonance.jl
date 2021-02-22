using AdaptiveResonance
using Test
using Logging
using DelimitedFiles
using Random

# Set the log level
LogLevel(Logging.Info)

# Auxiliary generic functions for loading data, etc.
include("test_utils.jl")

@testset "CVI.jl" begin
    @info "CVI Testing"
    # Parse the data
    data_file = "../data/full_data.csv"
    data = readdlm(data_file, ',')
    data = permutedims(data)
    train_x = data[1:2, :]
    train_y = data[3, :]

    # Create the ART module, train, and classify
    # art = SFAM()
    # train!(art, train_x, train_y)

    cvi = XB()
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

@testset "ARTSCENE.jl" begin
    # ARTSCENE training and testing
    include("test_artscene.jl")
end # @testset "ARTSCENE.jl"

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
