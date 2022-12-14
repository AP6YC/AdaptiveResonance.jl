"""
    common.jl

# Description
Tests of common code for the `AdaptiveResonance.jl` package.
"""

@testset "common" begin
    @info "------- Common Code Tests -------"
    # Example arrays
    three_by_two = [1 2; 3 4; 5 6]

    # Test DataConfig constructors
    @info "Testing DataConfig..."
    dc1 = DataConfig()                  # Default constructor
    dc2 = DataConfig(0, 1, 2)           # When min and max are same across all features
    dc3 = DataConfig([0, 1], [2, 3])    # When min and max differ across features
    dc4 = DataConfig(three_by_two)      # When a data matrix is provided

    # # Test get_n_samples
    # @info "Testing get_n_samples..."
    # @test get_n_samples([1,2,3]) == 1           # 1-D array case
    # @test get_n_samples(three_by_two) == 2      # 2-D array case

    # Test data_setup!
    @info "Testing data_setup!..."
    data_setup!(DDVFA(), three_by_two)
    data_setup!(DDVFA().config, three_by_two)

    # Test breaking situations
    @info "Testing common code error handling..."
    @test_throws ErrorException performance([1,2],[1,2,3])
    @test_logs (:warn,) AdaptiveResonance.data_setup!(dc3, three_by_two)
    bad_config =  DataConfig(1, 0, 3)
    @test_throws ErrorException linear_normalization(three_by_two, config=bad_config)
end # @testset "common.jl"

@testset "constants.jl" begin
    @info "------- Constants Tests -------"
    # Test that constants are exported
    art_constants = [
        ADAPTIVERESONANCE_VERSION,
        ART_MODULES,
        ARTMAP_MODULES,
        ADAPTIVERESONANCE_MODULES,
        DDVFA_METHODS,
        MATCH_FUNCTIONS,
        ACTIVATION_FUNCTIONS,
    ]
    for local_constant in art_constants
        @test @isdefined local_constant
    end
end

@testset "kwargs" begin
    @info "------- Kwargs test -------"

    # Iterate over all modules
    for art in ADAPTIVERESONANCE_MODULES
        art_module = art(alpha=1e-3, display=false)
    end
end

@testset "Incremental train!" begin
    # Create an FuzzyART module
    art = FuzzyART()

    # Create a small batch of data
    dim = 2
    n_samples = 3
    x = rand(dim, n_samples)

    # Setup the ART data config
    data_setup!(art, x)

    # Train incrementally before without batch operation
    for i = 1:n_samples
        train!(art, x)
    end
end