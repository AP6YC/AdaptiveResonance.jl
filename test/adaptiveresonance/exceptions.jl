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