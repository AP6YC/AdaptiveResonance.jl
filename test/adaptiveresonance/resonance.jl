"""
    resonance.jl

# Description
Resonance search test sets.
"""

# Shorthand to get low-level functions in tests
const AR = AdaptiveResonance

# Create a dummy data fixture for running tests
mutable struct SearchFixture <: ARTModule
    opts::NamedTuple
    threshold::Float64
    T::Vector{Float64}
    M::Vector{Float64}
    labels::Vector{Int}
    n_categories::Int
    stats::AR.ARTStats
end

AR.resonance_activation!(::SearchFixture, ::AR.RealVector) = nothing

@testset "resonance_saerch Usage" begin
    # Iterate over sorting strategies and activation/match results
    for sort in (false, true), scores in ([0.0, 0.0], [-1.0, -2.0], [0.8, 0.7])
        art = SearchFixture(
            (sort=sort,),
            0.5,
            copy(scores),
            [0.1, 0.8],
            [1, 2],
            2,
            AR.build_art_stats()
        )
        @test AR.resonance_search!(art, Float64[]) == (2, false)
        @test art.T == scores
        @test art.stats["T"] == scores[2]
        @test art.stats["M"] == 0.8

        art.M .= 0.1
        @test AR.resonance_search!(art, Float64[]) == (1, true)
        @test art.stats["bmu"] == 1
        @test art.stats["mismatch"]
        @test art.stats["T"] == scores[1]
        art.M .= 0.8

        # A supervisory conflict stops at the first resonant category.
        @test AR.resonance_search!(art, Float64[]; y=2) == (1, true)
    end

    @test_throws ArgumentError AR.resonance_search!(FuzzyART(), [0.0, 1.0])
end

@testset "Test Sort Strategy Equivalences" begin
    # Include factory variants, which must inherit the shared implementation.
    for constructor in (
        FuzzyART,
        DVFA,
        DDVFA,
        HypersphereART,
        SFAM,
        GammaNormalizedFuzzyART,
        DAM
    )
        a, b = constructor(sort=true), constructor(sort=false)
        for (x, y) in zip(
            (0.0, 1.0, 0.2, 0.8, 0.4, 0.6, 0.2),
            (1, 2, 1, 2, 2, 1, 2)
        )
            sample = a isa HypersphereART ? [x] : [x, 1-x]
            if a isa ARTMAP
                @test train!(a, sample, y; preprocessed=true) == train!(b, sample, y; preprocessed=true)
            else
                @test train!(a, sample; y=y, preprocessed=true) == train!(b, sample; y=y, preprocessed=true)
            end
            @test a.labels == b.labels
            @test a.n_categories == b.n_categories
            @test a.stats == b.stats
        end

        for x in 0.0:0.1:1.0, get_bmu in (false, true)
            # Temporary hack to determine if complement coding is needed, rewrite library to do this more elegantly
            sample = a isa HypersphereART ? [x] : [x, 1-x]
            @test classify(a, sample; preprocessed=true, get_bmu=get_bmu) == classify(b, sample; preprocessed=true, get_bmu=get_bmu)
            @test a.stats == b.stats
        end
    end
end

@testset "SFAM Match Tracking Visits Next Category" begin
    for sort in (false, true)
        art = SFAM(rho=0.5, sort=sort)
        train!(art, [0.8, 0.2], 1; preprocessed=true)
        train!(art, [0.7, 0.3], 2; preprocessed=true)

        # The first, smaller category has higher choice but lower match.
        art.W[:, 1] = [0.5, 0.0]
        @test train!(art, [0.8, 0.2], 2; preprocessed=true) == 2
        @test art.n_categories == 2
        @test art.stats["bmu"] == 2
        @test !art.stats["mismatch"]
    end
end

@testset "Test DVFA rho_lb Creates Category in New Cluster" begin
    for sort in (false, true)
        art = DVFA(rho_lb=0.4, rho_ub=0.9, sort=sort)
        train!(art, [0.0, 1.0]; preprocessed=true)
        @test train!(art, [0.5, 0.5]; preprocessed=true) == 1
        @test art.n_categories == 2
        @test art.n_clusters == 1
        @test !art.stats["mismatch"]
        @test train!(art, [1.0, 0.0]; preprocessed=true) == 1
    end
end
