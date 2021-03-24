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
    # data_file =
    train_x, train_y = get_cvi_data("../data/cvi/correct_partition.csv")
    n_samples = length(train_y)

    # Construct the cvis
    cvis = [
        XB(),
        DB(),
        PS()
    ]
    n_cvis = length(cvis)
    data_paths = readdir("../data/cvi", join=true)

    # SOLO
    @info "CVI Incremental Solo"
    data_path = "../data/cvi/over_partition.csv"
    @info "ICVI: Data $data_path"
    data, labels = get_cvi_data(data_path)
    cvi_i = deepcopy(cvis)
    for cvi in cvi_i
        @info "ICVI: $(typeof(cvi))"
        for ix = 1:n_samples
            # param_inc!(cvi, train_x[:, ix], train_y[ix])
            param_inc!(cvi, data[:, ix], labels[ix])
            evaluate!(cvi)
        end
    end

    # Incremental
    @info "CVI Incremental"
    cvi_i = deepcopy(cvis)
    for data_path in data_paths
        @info "ICVI: Data $data_path"
        data, labels = get_cvi_data(data_path)
        cvi_i = deepcopy(cvis)
        for cvi in cvi_i
            @info "ICVI: $(typeof(cvi))"
            for ix = 1:n_samples
                # param_inc!(cvi, train_x[:, ix], train_y[ix])
                param_inc!(cvi, data[:, ix], labels[ix])
                evaluate!(cvi)
            end
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

    # Incremental porcelain
    @info "CVI Incremental Porcelain"
    cvi_p = deepcopy(cvis)
    cvs_i = zeros(n_samples, n_cvis)
    for cx = 1:n_cvis
        for ix = 1:n_samples
            cvs_i[ix, cx] = get_icvi!(cvi_p[cx], train_x[:, ix], train_y[ix])
        end
    end

    # Batch porcelain
    @info "CVI Batch Porcelain"
    cvi_bp = deepcopy(cvis)
    cvs_b = zeros(n_cvis)
    for cx = 1:n_cvis
        cvs_b[cx] = get_cvi!(cvi_bp[cx], train_x, train_y)
    end

    # Test that the porcelain CV is the same as the others
    for cx = 1:n_cvis
        # Incremental
        @test isapprox(cvi_i[cx].criterion_value, cvs_i[end, cx])
        @test isapprox(cvi_b[cx].criterion_value, cvs_i[end, cx])

        # Batch
        @test isapprox(cvi_i[cx].criterion_value, cvs_b[cx])
        @test isapprox(cvi_b[cx].criterion_value, cvs_b[cx])
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
