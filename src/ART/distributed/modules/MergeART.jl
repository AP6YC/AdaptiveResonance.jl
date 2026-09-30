"""
    MergeART.jl

# Description
Includes all of the structures and logic for running a MergeART module.

# References
1. L. E. Brito da Silva, I. Elnabarawy, and D. C. Wunsch, 'Distributed dual vigilance fuzzy adaptive resonance theory learns online, retrieves arbitrarily-shaped clusters, and mitigates order dependence,' Neural Networks, vol. 121, pp. 208-228, 2020, doi: 10.1016/j.neunet.2019.08.033.
2. G. Carpenter, S. Grossberg, and D. Rosen, 'Fuzzy ART: Fast stable learning and categorization of analog patterns by an adaptive resonance system,' Neural Networks, vol. 4, no. 6, pp. 759-771, 1991.
"""


#region OPTIONS

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
    similarity::Symbol = :single; @assert similarity in LINKAGE_METHODS

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

#endregion


#region STRUCTS

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

#endregion


#region CONSTRUCTORS

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

#endregion

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
    return prototype_match(input_norm, weight_norm, score, reference)
end

"""
Convert a prototype activation into its normalized match value.

# Arguments
- `input_norm::Real`: norm of the incoming prototype.
- `weight_norm::Real`: norm of the candidate prototype.
- `score::Real`: previously computed activation.
- `reference::Real`: exponent for the destination-to-input norm ratio.

# Description

Reuses activation without recomputing the fuzzy intersection. Zero-norm inputs
retain the explicit convention used by `prototype_similarity`.
"""
function prototype_match(input_norm::Real, weight_norm::Real, score::Real, reference::Real)
    iszero(input_norm) && return iszero(weight_norm) ? 1.0 : 0.0
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
    method === :centroid && return centroid(art, destination, input, activation)
    # Keep both category axes: weighted linkage needs both sets of counts.
    scores = [prototype_similarity(
        art.opts,
        input.W[:, j],
        destination.W[:, i],
        activation,
        art.opts.gamma_ref
    ) for i in 1:destination.n_categories, j in 1:input.n_categories]
    # Weighted linkage needs both clusters; other reductions share the common API.
    return similarity(method, scores, destination, input)
end

"""
Compare the fuzzy envelopes of two local clusters.

# Arguments
- `art::MergeART`: model supplying activation and match parameters.
- `destination::FuzzyART`: candidate output cluster.
- `input::FuzzyART`: incoming cluster.
- `activation::Bool`: select activation instead of match.

# Description

Uses the componentwise minimum of each cluster's weights and preserves the
separate centroid match equation rather than reducing pairwise category scores.
"""
function centroid(art::MergeART, destination::FuzzyART, input::FuzzyART, activation::Bool)
    # The paper's centroid is a fuzzy envelope, not an arithmetic mean.
    left = cluster_envelope(destination)
    right = cluster_envelope(input)
    # Evaluate activation directly between the two cluster envelopes.
    activation && return prototype_similarity(art.opts, right, left, true, art.opts.gamma_ref)
    # For centroid matching, normalize their overlap by the incoming envelope norm.
    denominator = sum(right)
    iszero(denominator) && return iszero(sum(left)) ? 1.0 : 0.0
    return (sum(min.(left, right)) / denominator) ^ art.opts.gamma
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
"""
Evaluate a learned prototype against one local category.

# Arguments
- `art::FuzzyART`: local compression module.
- `input::MergePrototype`: learned weights, already in feature coordinates.
- `index::Integer`: destination category index.

# Description

Uses the prototype activation equation without treating the weights as a raw
complement-coded sample.
"""
function art_activation(art::FuzzyART, input::MergePrototype, index::Integer)
    return prototype_similarity(art.opts, input.weights, get_sample(art.W, index), true, 1.0)
end

"""
Evaluate a learned prototype's match with a local category.

# Arguments
- `art::FuzzyART`: local compression module.
- `input::MergePrototype`: incoming learned weights.
- `index::Integer`: destination category index.
- `score::Real`: optional cached activation for this comparison.

# Description

Normalizes by the incoming prototype's own norm with reference exponent one.
Supplying activation avoids repeating the fuzzy intersection and exponentiation.
"""
function art_match(art::FuzzyART, input::MergePrototype, index::Integer,
                   score::Real=art_activation(art, input, index))
    weight = get_sample(art.W, index)
    return prototype_match(sum(input.weights), sum(weight), score, 1.0)
end

"""
Compute activations and matches for local prototype compression.

# Arguments
- `art::FuzzyART`: local module whose search buffers are updated.
- `input::MergePrototype`: prototype presented for compression.

# Description

Evaluates each activation once, then supplies it to the matching API. The
resonance hooks delegate evaluation here and read the stored candidate matches.
"""
function activation_match!(art::FuzzyART, input::MergePrototype)
    # Prepare exactly one activation and match entry per destination category.
    resize!(art.T, art.n_categories)
    resize!(art.M, art.n_categories)
    for i in 1:art.n_categories
        art.T[i] = art_activation(art, input, i)
        art.M[i] = art_match(art, input, i, art.T[i])
    end
end

# Delegate prototype preparation to the same evaluation API used by sample inputs.
resonance_activation!(art::FuzzyART, input::MergePrototype) = activation_match!(art, input)
resonance_match!(art::FuzzyART, ::MergePrototype, bmu::Integer) = art.M[bmu]

# Cluster merging uses dimensionless lower vigilance; inference uses sample units.
resonance_threshold(art::MergeART, ::FuzzyART) = art.opts.rho_lb
resonance_threshold(art::MergeART, ::RealVector) = art.opts.rho_lb * art.config.dim
# Compression uses the local upper vigilance, independently of its sample threshold.
resonance_threshold(art::FuzzyART, ::MergePrototype) = art.opts.rho
# MergeART has no supervisory tracking option, including for callback searches.
resonance_match_tracking(::MergeART) = false

# Keep the stored threshold in sample units for inspection and ordinary inference.
function set_threshold!(art::MergeART)
    art.threshold = art.opts.rho_lb * art.config.dim
    return
end

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
            resonance_search!(result, MergePrototype(weights))
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
    # Validate and snapshot the source before replacing destination state.
    current = init_train!(source, art)
    initialize!(art, source)
    # Repeatedly merge the previous partition and compose its source mapping.
    for iteration in 1:art.opts.max_iter
        assignment = merge_pass!(art, current)
        art.source_map = assignment[art.source_map]
        art.epoch = iteration
        art.opts.display && @info "MergeART pass $iteration: $(art.n_categories) clusters"
        stopping_conditions(art, length(current)) && break
        current = art.F2
    end
    # Compress only after the cluster partition has finished merging.
    art.F2 = [compress_categories!(art, node) for node in art.F2]
    return copy(art.source_map)
end

"""
Validate and snapshot a DDVFA partition for MergeART training.

# Arguments
- `source::DDVFA`: trained source whose local clusters will be copied.
- `art::MergeART`: destination model for the prepared partition.

# Description

Checks prototype geometry and instance counts before destination state changes.
Returns independent local modules so subsequent merging preserves the source.
"""
function init_train!(source::DDVFA, art::MergeART)
    # Require a populated partition before examining its local modules.
    source.n_categories > 0 || throw(ArgumentError("MergeART requires a trained DDVFA."))
    for node in source.F2
        size(node.W, 1) == source.config.dim_comp || throw(DimensionMismatch("Incompatible prototype dimensions."))
        node.n_categories > 0 || throw(ArgumentError("Source clusters must be nonempty."))
        all(isfinite, node.W) && all(0 .<= node.W .<= 1) || throw(ArgumentError("Expected normalized fuzzy prototypes."))
        length(node.n_instance) == node.n_categories && all(node.n_instance .> 0) ||
            throw(ArgumentError("Each prototype requires a positive instance count."))
    end
    return deepcopy(source.F2)
end

"""
Initialize MergeART state for a prepared source partition.

# Arguments
- `art::MergeART`: destination model to reset.
- `source::DDVFA`: validated source supplying configuration and mapping size.

# Description

Copies the data configuration, clears the previous fit, and starts an identity
source mapping. Source validation is performed by `init_train!` before this step.
"""
function initialize!(art::MergeART, source::DDVFA)
    art.config = deepcopy(source.config)
    art.source_map = collect(1:source.n_categories)
    art.stats = build_art_stats()
    art.epoch = 0
    # Replace containers rather than clearing storage that a previous partition owns.
    art.F2 = FuzzyART[]
    art.labels = Int[]
    art.T = Float[]
    art.M = Float[]
    art.n_categories = 0
    set_threshold!(art)
    return
end

"""
Create a MergeART cluster from an incoming local module.

# Arguments
- `art::MergeART`: destination model to extend.
- `input::FuzzyART`: local module to copy without compressing its prototypes.
- `label::Integer`: identifier for the new output cluster.

# Description

Fast-commits independent weights and counts, updates the cluster state, and
returns the new cluster index.
"""
function create_category!(art::MergeART, input::FuzzyART, label::Integer)
    push!(art.F2, deepcopy(input))
    art.n_categories += 1
    push!(art.labels, label)
    return art.n_categories
end

"""
Learn an incoming cluster by concatenating its categories into the winner.

# Arguments
- `art::MergeART`: destination model performing cluster merging.
- `input::FuzzyART`: incoming local module and its instance counts.
- `bmu::Integer`: index of the resonant destination cluster.

# Description

Delegates to `merge_categories!`; prototype compression remains a separate stage.
"""
function learn!(art::MergeART, input::FuzzyART, bmu::Integer)
    merge_categories!(art.F2[bmu], input)
    return
end

"""
Assign one local FuzzyART module during a MergeART merging pass.

# Arguments
- `art::MergeART`: destination partition under construction.
- `input::FuzzyART`: prepared cluster in the same feature coordinates.

# Description

Returns the destination cluster index after resonance search and either learning
or category creation. This internal step does not update source provenance or
compress prototypes; the full DDVFA training method handles those operations.
"""
function train!(art::MergeART, input::FuzzyART)
    # The input-specific threshold selects lower vigilance in prototype-match units.
    isempty(art.F2) && return create_category!(art, input, 1)
    bmu, mismatch = resonance_search!(art, input)
    if mismatch
        bmu = create_category!(art, input, art.n_categories + 1)
    else
        learn!(art, input, bmu)
    end
    return bmu
end

"""
Build one fresh MergeART partition from the preceding partition.

# Arguments
- `art::MergeART`: destination model whose cluster storage is replaced.
- `partition::AbstractVector{FuzzyART}`: local modules to present in order.

# Description

Returns the input-cluster-to-output-cluster assignment for this pass. Fresh
containers prevent self-merging even when `partition` is the previous `art.F2`.
"""
function merge_pass!(art::MergeART, partition::AbstractVector{FuzzyART})
    # Detach the destination while preserving the previous partition as input.
    art.F2 = FuzzyART[]
    art.labels = Int[]
    art.T = Float[]
    art.M = Float[]
    art.n_categories = 0
    set_threshold!(art)
    assignment = zeros(Int, length(partition))
    for (i, node) in enumerate(partition)
        assignment[i] = train!(art, node)
    end
    return assignment
end

"""
Check whether MergeART has finished its merging passes.

# Arguments
- `art::MergeART`: model after completing the current pass.
- `previous_count::Integer`: number of clusters before that pass.

# Description

Passes only coarsen the partition. An unchanged cluster count therefore signals
unchanged membership; the iteration limit also ends merging.
"""
function stopping_conditions(art::MergeART, previous_count::Integer)
    return art.n_categories == previous_count || art.epoch >= art.opts.max_iter
end

# Reject raw training explicitly rather than entering ART's generic batch trainer.
function train!(::MergeART, ::RealVector; kwargs...)
    throw(ArgumentError("Train MergeART on a DDVFA model, not raw samples."))
end

function train!(::MergeART, ::RealMatrix; kwargs...)
    throw(ArgumentError("Train MergeART on a DDVFA model, not raw samples."))
end
