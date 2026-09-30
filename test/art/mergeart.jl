"""
    mergeart.jl

# Description
MergeART-specific test sets.
"""

# Short alias for testing internal helpers alongside the public API.
const AR = AdaptiveResonance

@info "------- MergeART Tests -------"

# Build an exact, independently counted source partition for analytical tests.
function merge_source(points; counts=ones(Int, length(points)))
    # Use exact-match vigilance to keep the analytical source prototypes separate.
    source = DDVFA(rho_lb=1.0, rho_ub=1.0, gamma=2.0, alpha=0.01)
    # Give each point a distinct label to force a new local module.
    for (i, x) in enumerate(points)
        train!(source, [x, 1-x]; y=i, preprocessed=true)
        # Assign a known count so later tests can check weighted linkage and count conservation.
        source.F2[end].n_instance[1] = counts[i]
    end
    return source
end

@testset "Construction Tests" begin
    # Reject an empty source and invalid merging parameters.
    @test_throws ArgumentError MergeART(DDVFA())
    @test_throws AssertionError MergeART(max_iter=0)
    @test_throws AssertionError MergeART(rho_lb=0.8, rho_ub=0.7)
    @test_throws AssertionError MergeART(similarity=:unknown)
    # Reject raw training inputs and inference before any source has been imported.
    @test_throws ArgumentError train!(MergeART(), [0.0, 1.0])
    @test_throws ArgumentError train!(MergeART(), [0.0 1.0])
    @test_throws ArgumentError classify(MergeART(), [0.0, 1.0]; preprocessed=true)
    # Keep this postprocessor outside the collection of raw-sample learners.
    @test !(MergeART in ART_MODULES)
    # Create two nearby clusters and a distant third, with ten instances in total.
    source = merge_source([0.0, 0.1, 1.0]; counts=[2, 3, 5])
    # Retain the original partition for checking that fitting leaves the source intact.
    original = deepcopy(source)
    model = MergeART(source; rho_lb=0.7, rho_ub=0.95)
    # Check parameter inheritance, cluster assignments, and preservation of sample counts.
    @test model.opts.alpha == source.opts.alpha
    @test model.n_categories == 2
    @test model.source_map == [1, 1, 2]
    @test sum(sum(n.n_instance) for n in model.F2) == 10
    # High upper vigilance keeps the two nearby prototypes distinct after cluster merging.
    @test model.F2[1].n_categories == 2
    # Verify source values are unchanged and mutable destination state is independently owned.
    @test source.labels == original.labels
    @test all(source.F2[i].W == original.F2[i].W for i in eachindex(source.F2))
    @test model.config !== source.config
    @test model.F2[2].W !== source.F2[3].W
    # Refitting imports a snapshot once, even after the source grows.
    first = deepcopy(model)
    @test train!(model, source) == first.source_map
    @test sum(sum(n.n_instance) for n in model.F2) == 10
    # Grow the source and rebuild to include the new cluster exactly once.
    train!(source, [0.9, 0.1]; y=4, preprocessed=true)
    train!(model, source)
    @test length(model.source_map) == 4
    @test sum(sum(n.n_instance) for n in model.F2) == 11
end

@testset "Pairwise linkage equations" begin
    # Compare a two-prototype destination with a one-prototype input.
    source = merge_source([0.0, 0.2, 0.8]; counts=[2, 3, 5])
    left = deepcopy(source.F2[1])
    AR.merge_categories!(left, source.F2[2])
    right = source.F2[3]
    # Check every supported linkage against manually computed scores.
    for method in LINKAGE_METHODS
        art = MergeART(alpha=0.01, gamma=2.0, similarity=method)
        # Both incoming and destination prototypes have norm one here.
        values = [(0.2 / 1.01)^2, (0.4 / 1.01)^2]
        # Counts two and three give destination probabilities 0.4 and 0.6 for weighted linkage.
        expected = method == :single ? maximum(values) :
                   method == :complete ? minimum(values) :
                   method == :weighted ? sum(values .* [0.4, 0.6]) : sum(values)/2
        # Centroid linkage uses an envelope of norm 0.8 rather than reducing the pairwise scores.
        if method == :centroid
            @test AR.similarity(art, left, right, true) ≈ (0.2/0.81)^2
            @test AR.similarity(art, left, right, false) ≈ 0.2^2
        else
            # Unit prototype norms make activation and match agree for the remaining methods.
            @test AR.similarity(art, left, right, true) ≈ expected
            @test AR.similarity(art, left, right, false) ≈ expected
        end
    end
    # A compressed input has a nonconstant norm: Equation 13 must use it.
    opts = opts_MergeART(alpha=0.01, gamma=2.0)
    @test AR.prototype_similarity(
        opts,
        [0.2, 0.4],
        [0.1, 0.4],
        false,
        1.0
    ) ≈ (0.5/0.6) * (0.5/0.51)^2
end

@testset "Compression and inference" begin
    # Exercise the full merge-and-compress workflow with both search strategies and all linkages.
    for sort in (false, true), method in LINKAGE_METHODS
        source = merge_source([0.0, 0.1, 1.0]; counts=[2, 3, 5])
        model = MergeART(source; rho_lb=0.7, rho_ub=0.7, similarity=method, sort=sort)
        # Verify the nearby clusters merge and their prototypes compress into one counted category.
        @test model.source_map == [1, 1, 2]
        @test model.F2[1].n_categories == 1
        @test model.F2[1].n_instance == [5]
        @test model.F2[1].W[:, 1] ≈ [0.0, 0.9]
        # Check single-sample and batch predictions against the two resulting clusters.
        @test classify(model, [0.0, 1.0]; preprocessed=true, get_bmu=true) == 1
        @test classify(model, [1.0, 0.0]; preprocessed=true, get_bmu=true) == 2
        @test classify(model, [0.0 1.0; 1.0 0.0]; preprocessed=true, get_bmu=true) == [1, 2]
        # Confirm the shared distributed utility counts compressed prototypes across clusters.
        @test AR.get_n_weights(model) == 2
    end
    # Zero envelopes are valid at zero vigilance and must not produce NaNs.
    source = merge_source([0.0, 1.0])
    model = MergeART(source; rho_lb=0.0, rho_ub=0.0)
    @test model.F2[1].W[:, 1] == [0.0, 0.0]
    @test model.F2[1].n_instance == [2]
    @test isfinite(AR.prototype_similarity(model.opts, zeros(2), zeros(2), false, 1.0))
end

@testset "Repeated passes and mapping composition" begin
    # Present the bridging point last so the first pass leaves two clusters.
    source = merge_source([0.0, 0.4, 0.2]; counts=[2, 3, 4])
    # Compare a single-pass limit with repeated merging until the partition stabilizes.
    once = MergeART(source; rho_lb=0.6, rho_ub=1.0, max_iter=1)
    stable = MergeART(source; rho_lb=0.6, rho_ub=1.0)
    # Check that mapping composition carries all original source nodes into the final cluster.
    @test once.source_map == [1, 2, 1]
    @test stable.source_map == [1, 1, 1]
    # Two passes merge the partition; the third confirms there is no further change.
    @test stable.epoch == 3
    # Upper vigilance one retains all three prototypes and their nine represented instances.
    @test stable.F2[1].n_categories == 3
    @test sum(stable.F2[1].n_instance) == 9
    @test length(stable.T) <= stable.n_categories
    # Raw inference uses a copied configuration from the source.
    raw = DDVFA(rho_lb=0.5, rho_ub=0.8)
    X = [10.0 11.0 19.0 20.0; 0.0 1.0 9.0 10.0]
    train!(raw, X)
    model = MergeART(raw)
    # Compare automatic preprocessing with explicitly complement-coded input.
    @test classify(model, X; get_bmu=true) ==
          classify(model, complement_code(X; config=model.config); preprocessed=true, get_bmu=true)
    # Ensure every source cluster has an entry in the final provenance mapping.
    @test length(model.source_map) == raw.n_categories
end

@testset "Lifecycle methods and shared linkage" begin
    source = merge_source([0.0, 0.1, 1.0]; counts=[2, 3, 5])
    model = MergeART(rho_lb=0.7, rho_ub=0.95, gamma=2.0, alpha=0.01)
    # Preparing a snapshot leaves the destination untouched and owns its weights.
    partition = AR.init_train!(source, model)
    @test model.n_categories == 0
    @test partition[1].W !== source.F2[1].W
    AR.initialize!(model, source)
    @test model.source_map == [1, 2, 3]
    @test model.epoch == 0
    @test model.config !== source.config
    # Incremental cluster training uses the same creation and learning dispatches.
    @test train!(model, partition[1]) == 1
    @test train!(model, partition[2]) == 1
    @test train!(model, partition[3]) == 2
    @test model.F2[1].n_instance == [2, 3]
    @test model.F2[1].W !== partition[1].W
    # A fresh pass may read the previous destination without modifying that input.
    previous = model.F2
    @test AR.merge_pass!(model, previous) == [1, 2]
    @test previous !== model.F2
    @test previous[1].n_instance == [2, 3]
    @test AR.stopping_conditions(model, 2)
    @test !AR.stopping_conditions(model, 3)
    model.epoch = model.opts.max_iter
    @test AR.stopping_conditions(model, 3)
    # Validate before initialization so invalid input does not discard a fitted model.
    @test_throws ArgumentError train!(model, DDVFA())
    @test model.n_categories == 2
    # Matrix reductions visit all pairs and match the corresponding vector result.
    scores = [0.1 0.9; 0.3 0.5]
    for method in (:single, :complete, :average, :median)
        @test AR.similarity(method, scores) == AR.similarity(method, vec(scores))
    end
    @test_throws ArgumentError AR.similarity(:unknown, scores)
    @test_throws ArgumentError AR.similarity(:unknown, source.F2[1], [0.0, 1.0], true)
    model.opts.similarity = :unknown
    @test_throws ArgumentError AR.similarity(model, partition[1], partition[2], true)
end

@testset "Prototype evaluation API and input thresholds" begin
    for sort in (false, true)
        source = DDVFA(rho_lb=1.0, rho_ub=1.0, gamma=2.0)
        train!(source, [0.0, 0.0, 1.0, 1.0]; y=1, preprocessed=true)
        train!(source, [1.0, 1.0, 0.0, 0.0]; y=2, preprocessed=true)
        model = MergeART(source; rho_lb=0.7, rho_ub=0.9, sort=sort)
        sample = [0.0, 0.0, 1.0, 1.0]
        node = model.F2[1]
        # Prototype evaluation reuses cached activation and matches the direct formula.
        for weights in (sample, [0.0, 0.0, 0.8, 0.7], zeros(4))
            input = AR.MergePrototype(weights)
            AR.activation_match!(node, input)
            for i in 1:node.n_categories
                score = AR.art_activation(node, input, i)
                @test node.T[i] == score
                @test node.M[i] ≈ AR.art_match(node, input, i)
                @test node.M[i] ≈ AR.art_match(node, input, i, score)
                @test node.M[i] ≈ AR.prototype_similarity(node.opts, weights, node.W[:, i], false, 1.0)
            end
        end
        # A dimension greater than one distinguishes cluster and sample units.
        @test AR.resonance_threshold(model, source.F2[1]) == 0.7
        @test AR.resonance_threshold(model, sample) == 1.4
        @test AR.resonance_threshold(node, AR.MergePrototype(sample)) == 0.9
        # Deliberately stale stored state must not control MergeART search behavior.
        model.threshold = 100.0
        node.threshold = 100.0
        @test AR.resonance_search!(model, source.F2[1]) == (1, false)
        @test classify(model, sample; preprocessed=true) == 1
        @test AR.resonance_search!(node, AR.MergePrototype(sample)) == (1, false)
        # Explicit numeric and callable thresholds still override the default.
        @test AR.resonance_search!(model, sample; threshold=100.0) == (1, true)
        @test AR.resonance_search!(model, sample; threshold=() -> 100.0) == (1, true)
        @test AR.resonance_search!(_ -> true, model, source.F2[1]) == (1, false)
        @test AR.resonance_search!(_ -> false, model, source.F2[1]) == (1, true)
        @test !AR.resonance_match_tracking(model)
        # Cluster training and sample inference can alternate without threshold repair.
        train!(model, source.F2[1])
        @test classify(model, sample; preprocessed=true) == 1
        @test model.threshold == 100.0
    end
end

@testset "Shared weighted and centroid linkage" begin
    source = merge_source([0.0, 0.2, 0.8, 1.0]; counts=[2, 3, 4, 1])
    left, right = deepcopy(source.F2[1]), deepcopy(source.F2[3])
    AR.merge_categories!(left, source.F2[2])
    AR.merge_categories!(right, source.F2[4])
    # Both axes have unequal probabilities, so each must contribute to weighting.
    @test AR.category_probabilities(left) ≈ [0.4, 0.6]
    @test AR.category_probabilities(right) ≈ [0.8, 0.2]
    scores = [0.1 0.9; 0.4 0.5]
    expected = 0.1*0.4*0.8 + 0.9*0.4*0.2 + 0.4*0.6*0.8 + 0.5*0.6*0.2
    @test AR.weighted(scores, left, right) ≈ expected
    @test AR.similarity(:weighted, scores, left, right) ≈ expected
    left.T = [0.1, 0.4]
    left.M = [0.3, 0.8]
    @test AR.weighted(left, true) ≈ 0.1*0.4 + 0.4*0.6
    @test AR.similarity(:weighted, left, [0.0, 1.0], false) ≈ 0.3*0.4 + 0.8*0.6
    for method in (:single, :complete, :average, :median)
        @test AR.similarity(method, scores, left, right) == AR.similarity(method, scores)
        @test AR.similarity(method, left.T, left) == AR.similarity(method, left.T)
    end
    @test_throws ArgumentError AR.similarity(:unknown, scores, left, right)
    @test_throws ArgumentError AR.similarity(:unknown, left.T, left)
    # Preserve the corrected exponent on the complete activation ratio.
    @test AR.cluster_envelope(left) ≈ [0.0, 0.8]
    sample = [0.4, 0.6]
    activation = (0.6 / (left.opts.alpha + 0.8)) ^ left.opts.gamma
    @test AR.centroid(left, sample, true) ≈ activation
    @test AR.centroid(left, sample, false) ≈ 0.8^left.opts.gamma_ref * activation
    @test LINKAGE_METHODS == [:single, :average, :complete, :median, :weighted, :centroid]
end
