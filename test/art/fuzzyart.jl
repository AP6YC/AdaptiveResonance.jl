"""
    fuzzyart.jl

# Description
FuzzyART-specific test sets.
"""

@testset "FuzzyART" begin
    @info "------- FuzzyART Test -------"

    # FuzzyART initialization and training
    my_FuzzyART = FuzzyART()
    train!(my_FuzzyART, data.train_x)

    # Compute a local sample for FuzzyART similarity method testing
    local_sample = complement_code(data.train_x[:, 1], config=my_FuzzyART.config)

    # Compute the local activation and match
    AdaptiveResonance.activation_match!(my_FuzzyART, local_sample)

    # Test that every method and field name computes
    for method in DDVFA_METHODS
        results = Dict()
        # for field_name in field_names
        for activation in (true, false)
            results[activation] = AdaptiveResonance.similarity(method, my_FuzzyART, local_sample, activation)
            # @test isapprox(truth[method][field_name], results[field_name])
        end
        @info "Method: $method" results
    end

    # Check the error handling of the similarity function
    # Access the wrong similarity metric keyword ("asdf")
    # @test_throws ErrorException AdaptiveResonance.similarity("asdf", my_FuzzyART, "T", local_sample)
    # Access the wrong output function ("A")
    # @test_throws ErrorException AdaptiveResonance.similarity("centroid", my_FuzzyART, "A", local_sample)
end