"""
    MergeART.jl

# Description
Includes all of the structures and logic for running a MergeART module.

# References
1. L. E. Brito da Silva, I. Elnabarawy, and D. C. Wunsch, 'Distributed dual vigilance fuzzy adaptive resonance theory learns online, retrieves arbitrarily-shaped clusters, and mitigates order dependence,' Neural Networks, vol. 121, pp. 208-228, 2020, doi: 10.1016/j.neunet.2019.08.033.
2. G. Carpenter, S. Grossberg, and D. Rosen, 'Fuzzy ART: Fast stable learning and categorization of analog patterns by an adaptive resonance system,' Neural Networks, vol. 4, no. 6, pp. 759-771, 1991.
"""

# -----------------------------------------------------------------------------
# TYPES
# -----------------------------------------------------------------------------

"""
MergeART options struct.

$(_OPTS_DOCSTRING)
"""
@with_kw mutable struct opts_MergeART <: ARTOpts @deftype Float
    """
    Lower-bound vigilance parameter: rho_lb ∈ [0, 1].
    """
    rho_lb = 0.7; @assert 0 <= rho_lb <= 1

    """
    Upper bound vigilance parameter: rho_ub ∈ [0, 1].
    """
    rho_ub = 0.85; @assert rho_lb <= rho_ub <= 1

    """
    Choice parameter: alpha > 0.
    """
    alpha = 1e-3; @assert isfinite(alpha) && alpha > 0

    """
    Learning parameter: beta ∈ (0, 1].
    """
    beta = 1.0; @assert 0 < beta <= 1

    """
    Pseudo kernel width: gamma >= 1.
    """
    gamma = 3.0; @assert isfinite(gamma) && gamma > 1

    """
    Reference gamma for normalization: 0 <= gamma_ref < gamma.
    """
    gamma_ref = 1.0; @assert 0 <= gamma_ref < gamma

    """
    Similarity method (activation and match): similarity ∈ [:single, :average, :complete, :median, :weighted, :centroid].
    """
    similarity::Symbol = :single; @assert similarity in DDVFA_METHODS

    """
    Maximum merging passes before final compression: max_iter ∈ [1, Inf).
    """
    max_iter::Int = 10; @assert max_iter >= 1

    """
    Flag to sort the F2 nodes by activation before the match phase

    When true, the F2 nodes are sorted by activation before match.
    When false, an iterative argmax and inhibition procedure is used to find the best-matching unit.
    """
    sort::Bool = false

    """
    Flag for verbose logging.
    """
    display::Bool = false
end

"""
Merge a DDVFA partition and compress the resulting local prototypes.

# Arguments
- `source::DDVFA`: optional trained source whose partition is copied and merged.
- `opts::opts_MergeART`: explicit options, or provide keyword options.

# Description

`MergeART(source; kwargs...)` inherits source vigilance, choice, learning, exponent, linkage, and sorting options unless overridden, then fits an independent model.
`MergeART(; kwargs...)` constructs an empty model.
`train!(model, source)` rebuilds from a snapshot rather than accumulating previously imported categories.
`source_map[i]` identifies the output cluster for source F2 node `i`; these are cluster identities, not supervisory labels. Sample inference uses `classify`.

# References
1. L. E. Brito da Silva, I. Elnabarawy, and D. C. Wunsch, 'Distributed dual vigilance fuzzy adaptive resonance theory learns online, retrieves arbitrarily-shaped clusters, and mitigates order dependence,' Neural Networks, vol. 121, pp. 208-228, 2020, doi: 10.1016/j.neunet.2019.08.033.
2. G. Carpenter, S. Grossberg, and D. Rosen, 'Fuzzy ART: Fast stable learning and categorization of analog patterns by an adaptive resonance system,' Neural Networks, vol. 4, no. 6, pp. 759-771, 1991.
"""
mutable struct MergeART <: DistributedART
    """
    MergeART options struct.
    """
    opts::opts_MergeART

    """
    Data configuration struct.
    """
    config::DataConfig

    # Working variables
    """
    Operating module threshold value, a function of the vigilance parameter.
    """
    threshold::Float

    """
    List of F2 nodes (themselves FuzzyART modules).
    """
    F2::Vector{FuzzyART}

    """
    Incremental list of labels corresponding to each F2 node, self-prescribed or supervised.
    """
    labels::ARTVector{Int}

    """
    Number of total categories.
    """
    n_categories::Int

    """
    Current training epoch.
    """
    epoch::Int

    """
    Activation values.
    """
    T::ARTVector{Float}

    """
    Match values.
    """
    M::ARTVector{Float}

    """
    Runtime statistics for the module, implemented as a dictionary containing entries at the end of each training iteration.
    These entries include the best-matching unit index and the activation and match values of the winning node.
    """
    stats::ARTStats

    """
    Mapping of original category labels to destination labels
    """
    source_map::Vector{Int}
end

# Validate keyword options before constructing an empty model.
MergeART(; kwargs...) = MergeART(opts_MergeART(; kwargs...))
function MergeART(opts::opts_MergeART)
    # Initialize cluster storage, search buffers, and provenance without importing data.
    return MergeART(
        opts,
        DataConfig(),
        opts.rho_lb,
        FuzzyART[],
        Int[],
        0,
        0,
        Float[],
        Float[],
        build_art_stats(),
        Int[]
    )
end

function MergeART(source::DDVFA; kwargs...)
    # Copy parameter values without sharing mutable options with the source.
    inherited = (; (name => getproperty(source.opts, name) for name in (
        :rho_lb,
        :rho_ub,
        :alpha,
        :beta,
        :gamma,
        :gamma_ref,
        :similarity,
        :sort
    ))...)
    # Explicit keyword arguments take precedence over inherited source parameters.
    art = MergeART(; merge(inherited, (; kwargs...))...)
    # Fit the independent destination from the source partition before returning it.
    train!(art, source)
    return art
end

"""
Compute a directed prototype-to-prototype activation or match.

# Arguments
- `opts`: options supplying alpha and gamma.
- `input::RealVector`: incoming prototype weights.
- `weight::RealVector`: destination prototype weights.
- `activation::Bool`: select activation instead of match.
- `reference::Real`: match normalization exponent.

# Description

Uses the incoming prototype norm. A zero-norm input has no remaining geometric extent and matches only another zero-norm prototype; this explicit limit avoids division by zero.
"""
function prototype_similarity(
    opts,
    input::RealVector,
    weight::RealVector,
    activation::Bool,
    reference::Real
)
    # Compute the fuzzy intersection and both prototype norms for the directed comparison.
    overlap = sum(min.(input, weight))
    weight_norm, input_norm = sum(weight), sum(input)
    # Normalize by the destination norm and apply the higher-order activation exponent.
    score = (overlap / (opts.alpha + weight_norm)) ^ opts.gamma
    # Activation needs no input-norm correction; match evaluation continues below.
    activation && return score
    # Handle a collapsed input before dividing by its norm.
    iszero(input_norm) && return iszero(weight_norm) ? 1.0 : 0.0
    # Rescale activation by the destination-to-input norm ratio to obtain the match.
    return (weight_norm / input_norm) ^ reference * score
end

"""
Compute linkage between two local FuzzyART clusters.

# Arguments
- `art::MergeART`: model supplying linkage and numerical parameters.
- `destination::FuzzyART`: candidate output cluster.
- `input::FuzzyART`: incoming cluster.
- `activation::Bool`: select activation instead of match.

# Description

Reduces pairwise prototype comparisons according to Table 3.
Weighted linkage uses instance probabilities from both clusters.
Centroid linkage compares the componentwise minimum envelopes, with its separate match equation.
"""
function similarity(
    art::MergeART,
    destination::FuzzyART,
    input::FuzzyART,
    activation::Bool
)
    # Select the reduction used to compare the incoming and destination clusters.
    method = art.opts.similarity
    if method === :centroid
        # The paper's centroid is a fuzzy envelope, not an arithmetic mean.
        left = vec(minimum(destination.W, dims=2))
        right = vec(minimum(input.W, dims=2))
        # Evaluate activation directly between the two cluster envelopes.
        activation && return prototype_similarity(art.opts, right, left, true, art.opts.gamma_ref)
        # For centroid matching, normalize their overlap by the incoming envelope norm.
        denominator = sum(right)
        iszero(denominator) && return iszero(sum(left)) ? 1.0 : 0.0
        return (sum(min.(left, right)) / denominator) ^ art.opts.gamma
    end
    # Keep both category axes: weighted linkage needs both sets of counts.
    scores = [prototype_similarity(
        art.opts,
        input.W[:, j],
        destination.W[:, i],
        activation,
        art.opts.gamma_ref
    ) for i in 1:destination.n_categories, j in 1:input.n_categories]
    # Reduce all pairwise scores for the unweighted linkage methods.
    method === :single && return maximum(scores)
    method === :complete && return minimum(scores)
    method === :average && return statistics_mean(scores)
    method === :median && return statistics_median(vec(scores))
    # For weighted linkage, turn counts into category probabilities in each cluster.
    p = destination.n_instance ./ sum(destination.n_instance)
    q = input.n_instance ./ sum(input.n_instance)
    # Weight each pair by the product of its two category probabilities.
    return sum(scores .* (p * q'))
end

# Cluster inputs reuse the same traversal and statistics as vector inputs.
function resonance_activation!(art::MergeART, input::FuzzyART)
    # Size the global search buffers to the current number of destination clusters.
    resize!(art.T, art.n_categories)
    resize!(art.M, art.n_categories)
    # Compute each candidate activation before the shared search orders the candidates.
    for i in 1:art.n_categories
        art.T[i] = similarity(art, art.F2[i], input, true)
    end
end
function resonance_match!(art::MergeART, input::FuzzyART, bmu::Integer)
    # Evaluate and store a match only when the search visits this destination cluster.
    art.M[bmu] = similarity(art, art.F2[bmu], input, false)
end

"""
Concatenate categories and counts into a destination cluster.

# Arguments
- `destination::FuzzyART`: cluster to extend.
- `input::FuzzyART`: source cluster whose weights and counts are copied.

# Description

Preserves every prototype during the merging stage; compression runs separately.
Local category labels are reassigned because they are not supervisory labels.
"""
function merge_categories!(destination::FuzzyART, input::FuzzyART)
    # Copy prototype columns and their associated counts in the same order.
    append!(destination.W, input.W)
    append!(destination.n_instance, input.n_instance)
    # Update the category total and assign contiguous local category identifiers.
    destination.n_categories += input.n_categories
    destination.labels = collect(1:destination.n_categories)
    return destination
end

# Distinguish a learned prototype from a raw, constant-norm input sample.
struct MergePrototype{V<:RealVector}
    weights::V
end
function resonance_activation!(art::FuzzyART, input::MergePrototype)
    # Prepare local search buffers and compare each prototype with reference exponent one.
    resize!(art.T, art.n_categories)
    resize!(art.M, art.n_categories)
    for i in 1:art.n_categories
        art.T[i] = prototype_similarity(art.opts, input.weights, art.W[:, i], true, 1.0)
        art.M[i] = prototype_similarity(art.opts, input.weights, art.W[:, i], false, 1.0)
    end
end
# Reuse the prototype matches computed alongside the local activations.
resonance_match!(art::FuzzyART, ::MergePrototype, bmu::Integer) = art.M[bmu]

"""
Compress a local cluster while preserving its total instance count.

# Arguments
- `art::MergeART`: model supplying upper vigilance and learning parameters.
- `input::FuzzyART`: cluster of prototypes to compress.

# Description

Returns a new FuzzyART cluster.
Each prototype is presented once with its stored count, using reference exponent one and its own norm in the match equation.
"""
function compress_categories!(art::MergeART, input::FuzzyART)
    # The resulting local module supports ordinary sample inference afterward.
    opts = opts_FuzzyART(
        rho=art.opts.rho_ub,
        alpha=art.opts.alpha,
        beta=art.opts.beta,
        gamma=art.opts.gamma,
        gamma_ref=1.0,
        gamma_normalization=true,
        uncommitted=false,
        sort=art.opts.sort
    )
    # Allocate a separate local module with the same feature configuration as the parent.
    result = FuzzyART(opts)
    result.config = deepcopy(art.config)
    result.W = ARTMatrix{Float}(undef, art.config.dim_comp, 0)
    # Set the local module threshold for subsequent ordinary sample inference.
    set_threshold!(result)
    # Present each stored prototype once, carrying the number of samples it represents.
    for i in 1:input.n_categories
        weights = input.W[:, i]
        count = input.n_instance[i]
        # An empty destination fast-commits the first prototype without search.
        bmu, mismatch = isempty(result.labels) ? (0, true) :
            resonance_search!(result, MergePrototype(weights); threshold=art.opts.rho_ub)
        if mismatch
            # Commit an unmatched prototype and replace the default count of one.
            create_category!(result, weights, result.n_categories + 1)
            result.n_instance[end] = count
        else
            # Update the resonant prototype and add the full incoming instance count.
            learn!(result, weights, bmu)
            result.n_instance[bmu] += count
        end
    end
    # Return the compressed cluster without modifying the incoming cluster.
    return result
end

"""
Rebuild MergeART from a snapshot of a trained DDVFA partition.

# Arguments
- `art::MergeART`: destination model to replace.
- `source::DDVFA`: trained source, which remains unchanged.

# Description

Merging passes use fresh destinations and compose `source_map` across passes.
Only after merging stops are prototypes compressed within each output cluster.
Returns the source-node-to-output-cluster mapping. Source supervisory labels are not constraints: MergeART is an unsupervised postprocessor.
"""
function train!(art::MergeART, source::DDVFA)
    # Require an existing partition before validating its individual local modules.
    source.n_categories > 0 || throw(ArgumentError("MergeART requires a trained DDVFA."))
    # Validate geometry and counts before replacing any destination state.
    for node in source.F2
        size(node.W, 1) == source.config.dim_comp || throw(DimensionMismatch("Incompatible prototype dimensions."))
        node.n_categories > 0 || throw(ArgumentError("Source clusters must be nonempty."))
        all(isfinite, node.W) && all(0 .<= node.W .<= 1) || throw(ArgumentError("Expected normalized fuzzy prototypes."))
        length(node.n_instance) == node.n_categories && all(node.n_instance .> 0) ||
            throw(ArgumentError("Each prototype requires a positive instance count."))
    end
    # Copy source state so neither merging nor compression mutates the source model.
    art.config = deepcopy(source.config)
    current = deepcopy(source.F2)
    # Start with the identity mapping and reset statistics from any previous fit.
    art.source_map = collect(1:source.n_categories)
    art.stats = build_art_stats()
    # Repeatedly merge the previous partition until it stabilizes or reaches the pass limit.
    for iteration in 1:art.opts.max_iter
        # Never search a cluster against itself or append it more than once.
        art.F2 = FuzzyART[]
        art.labels = Int[]
        art.T = Float[]
        art.M = Float[]
        art.n_categories = 0
        art.threshold = art.opts.rho_lb
        # Record where each input cluster lands during this particular pass.
        assignment = zeros(Int, length(current))
        for (i, node) in enumerate(current)
            # The first cluster seeds the destination; later clusters use resonance search.
            bmu, mismatch = isempty(art.F2) ? (0, true) : resonance_search!(art, node)
            if mismatch
                # Fast-commit an independent copy when no destination cluster resonates.
                push!(art.F2, deepcopy(node))
                art.n_categories += 1
                push!(art.labels, art.n_categories)
                bmu = art.n_categories
            else
                # Preserve all incoming prototypes inside the resonant cluster until compression.
                merge_categories!(art.F2[bmu], node)
            end
            # Save the destination index for composing the original source mapping.
            assignment[i] = bmu
        end
        # Compose provenance back to the original source cluster indices.
        art.source_map = assignment[art.source_map]
        art.epoch = iteration
        art.opts.display && @info "MergeART pass $iteration: $(art.n_categories) clusters"
        # Passes only coarsen the partition; no reduction means no membership change.
        art.n_categories == length(current) && break
        # Use the completed partition as input to the next fresh merging pass.
        current = art.F2
    end
    # After all merging passes, compress prototypes separately within each final cluster.
    art.F2 = [compress_categories!(art, node) for node in art.F2]
    # Sample matches use the existing dimension-scaled distributed inference path.
    art.threshold = art.opts.rho_lb * art.config.dim
    # Return an independent mapping so callers cannot mutate the stored provenance.
    return copy(art.source_map)
end

# Reject raw training explicitly rather than entering ART's generic batch trainer.
function train!(::MergeART, ::RealVector; kwargs...)
    throw(ArgumentError("Train MergeART on a DDVFA model, not raw samples."))
end

function train!(::MergeART, ::RealMatrix; kwargs...)
    throw(ArgumentError("Train MergeART on a DDVFA model, not raw samples."))
end
