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
            (sort=sort, match_tracking=false, epsilon=0.001),
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

@testset "Shared match tracking" begin
    for sort in (false, true)
        art = SearchFixture((sort=sort, match_tracking=true, epsilon=0.01),
                            0.5, [0.9, 0.8, 0.7], [0.6, 0.605, 0.8],
                            [1, 2, 2], 3, AR.build_art_stats())
        # The first label conflict raises vigilance; the next correct label is
        # still rejected because its match does not clear the raised threshold.
        @test AR.resonance_search!(art, Float64[]; y=2) == (3, false)
        @test art.threshold == 0.5
        @test art.T == [0.9, 0.8, 0.7]
        # A new sample starts from baseline, and unsupervised search never tracks.
        @test AR.resonance_search!(art, Float64[]; y=1) == (1, false)
        @test AR.resonance_search!(art, Float64[]) == (1, false)
        # Repeated conflicts can exhaust the search without changing the baseline.
        art.M .= [0.6, 0.7, 0.705]
        art.labels .= [1, 1, 2]
        @test AR.resonance_search!(art, Float64[]; y=2) == (1, true)
        @test art.stats["T"] == 0.9
        @test art.threshold == 0.5
        # Disabling tracking preserves the immediate supervisory mismatch.
        art.opts = (sort=sort, match_tracking=false, epsilon=0.01)
        @test AR.resonance_search!(art, Float64[]; y=2) == (1, true)
    end
end

@testset "Tracking options and module integration" begin
    for constructor in (FuzzyART, DVFA, DDVFA, HypersphereART, SFAM,
                        GammaNormalizedFuzzyART, DAM)
        @test constructor().opts.match_tracking == (constructor in (SFAM, DAM))
        @test_throws AssertionError constructor(epsilon=0.0)
        @test_throws AssertionError constructor(epsilon=Inf)
        for sort in (false, true)
            art = constructor(sort=sort, match_tracking=true)
            sample = art isa HypersphereART ? [0.2] : [0.2, 0.8]
            # Conflicting exact matches force category creation even with tracking.
            if art isa ARTMAP
                train!(art, sample, 1; preprocessed=true)
                train!(art, sample, 2; preprocessed=true)
                @test train!(art, sample, 2; preprocessed=true) == 2
            else
                train!(art, sample; y=1, preprocessed=true)
                train!(art, sample; y=2, preprocessed=true)
                @test train!(art, sample; y=2, preprocessed=true) == 2
            end
            @test art.n_categories == 3
            @test art.stats["mismatch"]
            # Inference remains label-free and starts at baseline vigilance.
            @test classify(art, sample; preprocessed=true) == 1
            @test !art.stats["mismatch"]
        end
    end
    # Match increments follow the threshold's scale, including gamma variants.
    for art in (FuzzyART(gamma_normalization=true, gamma_ref=2.0),
                DDVFA(gamma_ref=2.0), DVFA())
        train!(art, [0.2, 0.3, 0.8, 0.7]; preprocessed=true)
        @test AR.resonance_match_scale(art) == (art isa DVFA ? 2 : 4)
    end
end

@testset "SFAM tracking toggle and zero label" begin
    for sort in (false, true), tracking in (false, true)
        art = SFAM(rho=0.5, sort=sort, match_tracking=tracking)
        train!(art, [0.8, 0.2], 1; preprocessed=true)
        train!(art, [0.7, 0.3], 0; preprocessed=true)
        art.W[:, 1] = [0.5, 0.0]
        @test train!(art, [0.8, 0.2], 0; preprocessed=true) == 0
        @test art.n_categories == (tracking ? 2 : 3)
        @test art.stats["mismatch"] == !tracking
        @test art.opts.rho == 0.5
    end
end

@testset "DVFA lower-vigilance label conflicts" begin
    for sort in (false, true), tracking in (false, true)
        art = DVFA(rho_lb=0.4, rho_ub=0.9, sort=sort, match_tracking=tracking)
        train!(art, [0.0, 1.0]; y=1, preprocessed=true)
        train!(art, [1.0, 0.0]; y=2, preprocessed=true)
        # Both categories match at the lower bound. Tracking rejects the first
        # label and raises vigilance above the tied second match.
        @test train!(art, [0.5, 0.5]; y=2, preprocessed=true) == 2
        @test art.stats["mismatch"] == tracking
        @test art.n_clusters == (tracking ? 3 : 2)
    end
end

@testset "Supervised ART can learn a later resonant category" begin
    for sort in (false, true), tracking in (false, true)
        art = FuzzyART(rho=0.4, sort=sort, match_tracking=tracking)
        train!(art, [0.8, 0.2]; y=1, preprocessed=true)
        train!(art, [0.7, 0.3]; y=2, preprocessed=true)
        # The incorrect small category wins choice, but the correct category
        # clears its tracked match and should learn without adding a category.
        art.W[:, 1] = [0.5, 0.0]
        @test train!(art, [0.8, 0.2]; y=2, preprocessed=true) == 2
        @test art.n_categories == (tracking ? 2 : 3)
        @test art.stats["mismatch"] == !tracking
        @test art.threshold == 0.4
        if tracking
            @test art.W[:, 2] ≈ [0.7, 0.2]
            @test art.n_instance[2] == 2
        end
    end
end
