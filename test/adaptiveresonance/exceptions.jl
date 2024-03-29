"""
    exceptions.jl

# Description
Tests the edge cases and exceptions of the entire `AdaptiveResonance.jl` package.
"""

# Mismatch testset
# Enough ART modules do not encounter mismatch during the normal traing routines that these can be tested together.
@testset "Mismatch" begin
    @info "------- Mismatch test -------"

    # ART
    arts = [art(display=false) for art in ART_MODULES]
    artmaps = [artmap(display=false) for artmap in ARTMAP_MODULES]

    # Train on disparate data
    local_data = [
        0.0 1.0;
        0.0 1.0;
    ]
    local_labels= [1, 1]

    # Test on data that is still within range but equally far from other points
    test_data = [0.5, 0.5]

    # Get mismatch in unsupervised ART modules
    for art in arts
        train!(art, local_data)
        classify(art, test_data)
    end

    # Get mismatch in supervised ARTMAP modules
    for artmap in artmaps
        train!(artmap, local_data, local_labels)
        classify(artmap, test_data)
    end
end

@testset "init_tain!" begin
    # Create a new FuzzyART module
    art = FuzzyART()

    # Test that the new module's data config is not setup
    @test art.config.setup == false

    # Test that initializing training fails if the data is not preprocessed
    # and the data config is not setup (using the RealVector function)
    x = rand(2)
    @test_throws ErrorException AdaptiveResonance.init_train!(x, art, false)

    # Create faulty data and say that it is preprocessed
    x_cc_bad = rand(3)
    @test_throws ErrorException AdaptiveResonance.init_train!(x_cc_bad, art, true)
end

@testset "init_classify!" begin
    # Create a new FuzzyART module
    art = FuzzyART()

    # Test that the new module's data config is not setup
    @test art.config.setup == false

    # Test that initializing classification fails if the data is not
    # preprocessed and the data config is not setup
    @test_throws ErrorException AdaptiveResonance.init_classify!(rand(2, 2), art, false)
end
