using AdaptiveResonance
using Test
using Logging
using DelimitedFiles

# Set the log level
LogLevel(Logging.Info)

# Auxiliary generic functions for loading data, etc.
include("test_utils.jl")

# Load the data and test across all supervised modules
data = load_iris("../data/Iris.csv")

@testset "common.jl" begin
    @info "------- Common Code Tests -------"
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
end # @testset "common.jl"

@testset "constants.jl" begin
    @info "------- Constants Tests -------"
    ddvfa_methods = [
        "single",
        "average",
        "complete",
        "median",
        "weighted",
        "centroid"
    ]
    @test AdaptiveResonance.DDVFA_METHODS == ddvfa_methods
end # @testset "constants.jl"

@testset "AdaptiveResonance.jl" begin
    # Module loading
    include("modules.jl")
end # @testset "AdaptiveResonance.jl"

@testset "Train Test" begin
    # All ART modules
    arts = [
        DVFA,
        DDVFA
    ]
    n_arts = length(arts)

    # All common ART options
    art_opts = [
        (display = true,),
        (display = false,),
    ]
    n_art_opts = length(art_opts)

    # All test option permutations
    test_opts = [
        (get_bmu = true,),
        (get_bmu = false,)
    ]
    n_test_opts = length(test_opts)

    @info "--------- TRAIN TEST ---------"
    # ART
    perf_baseline = 0.8

    # Iterate over all ART modules
    for ix = 1:n_arts
        # Iterate over all test options
        for jx = 1:n_test_opts
            for kx = 1:n_art_opts
                # Unsupervised
                train_test_art(arts[ix](;art_opts[kx]...), data; test_opts=test_opts[jx])

                # Supervised
                @test train_test_art(arts[ix](;art_opts[kx]...), data; supervised=true, test_opts=test_opts[jx]) >= perf_baseline
            end
        end
    end

    @info "--------- END TRAIN TEST ---------"
end

@testset "kwargs" begin
    @info "--------- KWARGS TEST ---------"

    arts = [
        DVFA,
        DDVFA,
        SFAM,
        DAM
    ]

    for art in arts
        art_module = art(alpha=1e-3, display=false)
    end

    @info "--------- END KWARGS TEST ---------"
end

@testset "DVFA.jl" begin
    @info "------- DVFA Unsupervised -------"

    # Train and classify
    art = DVFA()
    y_hat_train = train!(art, data.train_x)

    @info "------- DVFA Supervised -------"

    # Train and classify
    art = DVFA()
    y_hat_train = train!(art, data.train_x, y=data.train_y)
    y_hat = classify(art, data.test_x)
    y_hat_bmu = classify(art, data.test_x, get_bmu=true)

    # Calculate performance
    perf_train = performance(y_hat_train, data.train_y)
    perf_test = performance(y_hat, data.test_y)
    perf_test_bmu = performance(y_hat_bmu, data.test_y)

    # Test the performances are above a baseline
    perf_baseline = 0.8
    @test perf_train >= perf_baseline
    @test perf_test >= perf_baseline
    @test perf_test_bmu >= perf_baseline
    @info art.n_categories

    # Log the results
    @info "DVFA Training Perf: $perf_train"
    @info "DVFA Testing Perf: $perf_test"
    @info "DVFA Testing BMU Perf: $perf_test_bmu"
end

@testset "DDVFA.jl" begin
    # DDVFA training and testing
    include("test_ddvfa.jl")
end # @testset "DDVFA.jl"

@testset "ARTMAP.jl" begin
    # Declare the baseline performance for all modules
    perf_baseline = 0.7

    # Iterate over each artmap module
    for art in [SFAM, DAM]
        perf = train_test_artmap(art(), data)
        @test perf >= perf_baseline
    end
end # @testset "ARTMAP.jl"

@testset "ARTSCENE.jl" begin
    # ARTSCENE training and testing
    include("test_artscene.jl")
end # @testset "ARTSCENE.jl"
