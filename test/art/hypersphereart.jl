"""
    hypersphereart.jl

# Description
HypersphereART-specific test sets.
"""

# Short alias for testing internal helpers alongside the public API.
const AR = AdaptiveResonance


@info "------- HypersphereART Tests -------"

@testset "Options and constructors" begin
    # Explicit options should be retained by an initially empty, registered learner.
    opts = opts_HypersphereART(rho=0.8, r_bar=2.0)
    art = HypersphereART(opts)
    @test art.opts === opts
    @test art isa ART
    @test HypersphereART in ART_MODULES
    @test isempty(art.W)
    @test !art.config.setup

    # Reject parameter values outside the supported ranges.
    for kwargs in (
        (rho=-0.1,),
        (rho=1.1,),
        (alpha=0.0,),
        (alpha=Inf,),
        (beta=0.0,),
        (beta=1.1,),
        (r_bar=0.0,),
        (r_bar=Inf,),
        (max_epoch=0,),
    )
        @test_throws AssertionError opts_HypersphereART(; kwargs...)
    end
end

@testset "Sphere geometry" begin
    # Check both fast learning and a partial update toward the same sample.
    for beta in (1.0, 0.5)
        art = HypersphereART(rho=0.5, alpha=0.1, beta=beta, r_bar=2.0)
        # The first sample creates a point sphere at the origin.
        @test train!(art, [0.0, 0.0], preprocessed=true) == 1
        @test art.W[:, 1] == [0.0, 0.0, 0.0]

        # A sample at unit distance has a known activation and lies on vigilance.
        @test AR.art_activation(art, [0.6, 0.8], 1) ≈ 1 / 2.1
        @test AR.art_match(art, [0.6, 0.8], 1) ≈ 0.5

        # Center displacement and radius growth each scale with the learning rate.
        @test train!(art, [0.6, 0.8], preprocessed=true) == 1
        @test art.W[:, 1] ≈ beta .* [0.3, 0.4, 0.5]
        @test art.n_instance == [2]
        previous = copy(art.W)

        # Repeated center, interior, and boundary points must not expand a sphere.
        for point in (beta .* [0.3, 0.4], beta .* [0.15, 0.2], [0.0, 0.0])
            train!(art, point, preprocessed=true)
            @test art.W ≈ previous
            @test all(isfinite, art.W)
        end

        # Shared in-place evaluation should agree with individual symbol calls.
        AR.activation_match!(art, [0.2, 0.3])
        @test art.T[1] ≈ AR.art_activation(art, [0.2, 0.3], 1)
        @test art.M[1] ≈ AR.art_match(art, [0.2, 0.3], 1)
    end
end

@testset "Batch and incremental preprocessing" begin
    # Include unequal feature ranges and a constant row to exercise normalization.
    x = [
        10.0 12.0 20.0;
        5.0 5.0 5.0;
        -2.0 0.0 8.0
    ]
    normalized = [
        0.0 0.2 1.0;
        0.0 0.0 0.0;
        0.0 0.2 1.0
    ]
    # Compare raw batch, configured incremental, and already-normalized training.
    original = copy(x)
    batch, incremental, prepared = HypersphereART(), HypersphereART(), HypersphereART()
    labels = train!(batch, x)
    data_setup!(incremental, x)
    @test [train!(incremental, x[:, j]) for j in axes(x, 2)] == labels
    @test train!(prepared, normalized, preprocessed=true) == labels
    @test batch.W ≈ incremental.W ≈ prepared.W

    # Preprocessing must preserve input data and store only center-plus-radius weights.
    @test x == original
    @test batch.config.dim == 3
    @test size(batch.W, 1) == 4
    @test batch.r_bar ≈ sqrt(3) / 2
    @test classify(batch, x) == classify(prepared, normalized, preprocessed=true)

    # Preprocessed odd-dimensional vectors retain all original features.
    art = HypersphereART()
    train!(art, [0.1, 0.2, 0.3], preprocessed=true)
    @test art.config.dim == 3
    @test art.W[1:3, 1] == [0.1, 0.2, 0.3]

    # Entirely constant data normalizes to one point and forms a single category.
    constant = HypersphereART()
    @test train!(constant, fill(7.0, 2, 3)) == [1, 1, 1]
    @test all(iszero, constant.W)

    # Repeated epochs should revisit each exact-match category once per epoch.
    epochs = HypersphereART(max_epoch=3, rho=1.0)
    train!(epochs, normalized, preprocessed=true)
    @test epochs.epoch == 3
    @test epochs.n_instance == [3, 3, 3]
end

@testset "Vigilance, search, labels, and statistics" begin
    # Exercise both sorted search and iterative activation inhibition.
    for sort in (false, true)
        # Exact-match vigilance separates distinct samples while preserving labels.
        art = HypersphereART(rho=1.0, r_bar=1.0, sort=sort)
        @test train!(art, [0.0], y=10, preprocessed=true) == 10
        @test train!(art, [1.0], y=20, preprocessed=true) == 20
        @test train!(art, [0.0], y=10, preprocessed=true) == 10

        # An existing but conflicting label forces a new category.
        @test train!(art, [0.0], y=20, preprocessed=true) == 20
        @test art.labels == [10, 20, 20]
        @test art.stats["mismatch"]

        # A distant query is rejected, but get_bmu can return its strongest category.
        weights = copy(art.W)
        counts = copy(art.n_instance)
        @test classify(art, [3.0], preprocessed=true) == -1
        @test art.stats["bmu"] == 2
        @test art.stats["T"] < 0
        @test classify(art, [3.0], preprocessed=true, get_bmu=true) == 20

        # Classification may update statistics but must not learn or count instances.
        @test art.W == weights
        @test art.n_instance == counts
        search = HypersphereART(rho=0.0, r_bar=1.0, sort=sort)
        train!(search, [0.0], preprocessed=true)
        train!(search, [1.0], y=2, preprocessed=true)

        # Exercise an actual multi-category vigilance search, without changing W.
        search.W[:, 1] = [0.0, 0.9]
        search.W[:, 2] = [0.8, 0.0]
        search.opts.rho = 0.3
        @test classify(search, [0.6], preprocessed=true) == 2
    end

    # Both search strategies should produce the same learned weights and predictions.
    x = [0.0 0.1 0.5 0.6 1.0]
    a, b = HypersphereART(sort=true), HypersphereART(sort=false)
    @test train!(a, x) == train!(b, x)
    @test a.W == b.W
    @test classify(a, x) == classify(b, x)
end

@testset "Invalid inputs" begin
    # Raw incremental training needs bounds, and classification needs a trained model.
    @test_throws ErrorException train!(HypersphereART(), [1.0])
    @test_throws ErrorException classify(HypersphereART(), [1.0], preprocessed=true)
    art = HypersphereART()
    train!(art, [0.0, 0.0], preprocessed=true)

    # Configured models reject inconsistent feature counts in either workflow.
    @test_throws DimensionMismatch train!(art, [0.0], preprocessed=true)
    @test_throws DimensionMismatch classify(art, zeros(3, 2), preprocessed=true)

    # Reject NaN and Inf before they can enter geometry calculations.
    @test_throws ArgumentError train!(art, [NaN, 0.0], preprocessed=true)
    @test_throws ArgumentError classify(art, [Inf, 0.0], preprocessed=true)
end

@testset "Shared symbol dispatch" begin
    # Custom hooks verify that the shared dispatch honors the option symbols,
    # rather than bypassing them through HypersphereART-specific methods.
    @eval AdaptiveResonance begin
        test_hypersphere_activation(art, x, W) = 0.75
        test_hypersphere_match(art, x, W) = 1.0
        test_hypersphere_update(art, x, W) = copy(W)
    end

    # Select the custom hooks through the same option symbols used by production code.
    art = HypersphereART(
        activation=:test_hypersphere_activation,
        match=:test_hypersphere_match,
        update=:test_hypersphere_update,
    )
    train!(art, [0.0, 0.0], preprocessed=true)

    # Forced resonance uses the custom scores while the custom update keeps W fixed.
    initial = copy(art.W)
    @test train!(art, [1.0, 1.0], preprocessed=true) == 1
    @test art.T == [0.75]
    @test art.M == [1.0]
    @test art.W == initial
    @test art.n_instance == [2]
    @test classify(art, [1.0, 1.0], preprocessed=true) == 1

    # The default update is functional, including when given a view of W.
    default = HypersphereART(r_bar=2.0)
    train!(default, [0.0, 0.0], preprocessed=true)
    weight = view(default.W, :, 1)
    updated = AR.hypersphere_update(default, [0.6, 0.8], weight)
    @test updated ≈ [0.3, 0.4, 0.5]
    @test weight == [0.0, 0.0, 0.0]
end
