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

    # Construct the cvis
    cvis = [
        XB(),
        DB()
    ]
    n_cvis = length(cvis)

    # Incremental
    @info "CVI Incremental"
    cvi_i = deepcopy(cvis)
    for cvi in cvi_i
        for ix = 1:n_samples
            param_inc!(cvi, train_x[:, ix], train_y[ix])
            evaluate!(cvi)
        end
    end

    # Batch
    @info "CVI Batch"
    cvi_b = deepcopy(cvis)
    for cvi in cvi_b
        param_batch!(cvi, train_x, train_y)
        evaluate!(cvi)
    end

    # Test that the criterion values are the same
    for i = 1:n_cvis
        @test isapprox(cvi_i[i].criterion_value, cvi_b[i].criterion_value)
    end

    # Porcelain
    @info "CVI Incremental Porcelain"
    cvi_p = deepcopy(cvis)
    cvs = zeros(n_samples, n_cvis)
    for cx = 1:n_cvis
        for ix = 1:n_samples
            cvs[ix, cx] = get_icvi(cvi_p[cx], train_x[:, ix], train_y[ix])
        end
    end

    # Test that the porcelain CV is the same as the others
    for cx = 1:n_cvis
        @test isapprox(cvi_i[cx].criterion_value, cvs[end, cx])
        @test isapprox(cvi_b[cx].criterion_value, cvs[end, cx])
    end
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
