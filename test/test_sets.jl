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
    data_file = "../data/correct_partition.csv"
    data = readdlm(data_file, ',')
    data = permutedims(data)
    train_x = data[1:2, :]
    train_y = convert(Array{Int64}, data[3, :])
    n_samples = length(train_y)

    # Incremental
    @info "CVI Incremental"
    cvi_i = XB()
    for ix = 1:n_samples
        param_inc!(cvi_i, train_x[:, ix], train_y[ix])
        evaluate!(cvi_i)
    end

    # Batch
    @info "CVI Batch"
    cvi_b = XB()
    vind = param_batch!(cvi_b, train_x, train_y)
    evaluate!(cvi_b)

    # Test that the criterion values are the same
    @test isapprox(cvi_i.criterion_value, cvi_b.criterion_value)

    # Test the porcelain functions
    @info "CVI Incremental Porcelain"
    cvi_p = XB()
    cvs = zeros(n_samples)
    for ix = 1:n_samples
        cvs[ix] = get_icvi(cvi_i, train_x[:, ix], train_y[ix])
    end

    # Test that the porcelain CV is the same as the others
    @test isapprox(cvi_i.criterion_value, cvs[end])
    @test isapprox(cvi_b.criterion_value, cvs[end])
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
