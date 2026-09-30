"""
    common.jl

# Description
Contains all common code for distributed ART modules, such as DDVFA.
"""

"""Abstract supertype for ART models whose clusters are local FuzzyART modules."""
abstract type DistributedART <: ART end

# COMMON DOC: Distributed ART incremental classification method
function classify(art::DistributedART, x::RealVector ; preprocessed::Bool=false, get_bmu::Bool=false)
    # Preprocess the data
    sample = init_classify!(x, art, preprocessed)

    bmu, mismatch = resonance_search!(art, sample)
    return mismatch && !get_bmu ? -1 : art.labels[bmu]
end

"""
Convenience function; return a concatenated array of all distributed ART weights.

# Arguments
- `art::DistributedART`: the distributed module to get all of the weights from as a list.
"""
function get_W(art::DistributedART)
    # Return a concatenated array of the weights
    return [art.F2[kx].W for kx = 1:art.n_categories]
end

"""
Convenience function; return the number of weights in each category as a vector.

# Arguments
- `art::DistributedART`: the distributed module to get all of the weights from as a list.
"""
function get_n_weights_vec(art::DistributedART)
    return [art.F2[i].n_categories for i = 1:art.n_categories]
end

"""
Convenience function; return the sum total number of weights in the distributed module.
"""
function get_n_weights(art::DistributedART)
    # Return the number of weights across all categories
    return sum(get_n_weights_vec(art))
end

# Distributed activation uses each local module's linkage; global matches are lazy.
function resonance_activation!(art::DistributedART, sample::RealVector)
    accommodate_vector!(art.T, art.n_categories)
    accommodate_vector!(art.M, art.n_categories)
    for j in 1:art.n_categories
        activation_match!(art.F2[j], sample)
        art.T[j] = similarity(art.opts.similarity, art.F2[j], sample, true)
    end
end

function resonance_match!(art::DistributedART, sample::RealVector, bmu::Integer)
    art.M[bmu] = similarity(art.opts.similarity, art.F2[bmu], sample, false)
end

# Argument docstring for the F2 field, includes the argument header
const FIELD_DOCSTRING = """
# Arguments
- `field::RealArray`: the activation or match scores to reduce across all entries.
"""

"""
Single linkage distributed ART similarity function.

$FIELD_DOCSTRING
"""
function single(field::RealArray)
    return maximum(field)
end

"""
Average linkage distributed ART similarity function.

$FIELD_DOCSTRING
"""
function average(field::RealArray)
    return statistics_mean(field)
end

"""
Complete linkage distributed ART similarity function.

$FIELD_DOCSTRING
"""
function complete(field::RealArray)
    return minimum(field)
end

"""
Median linkage distributed ART similarity function.

$FIELD_DOCSTRING
"""
function median(field::RealArray)
    return statistics_median(field)
end

"""
Reduce activation or match scores using an unweighted linkage method.

# Arguments
- `method::Symbol`: one of `:single`, `:complete`, `:average`, or `:median`.
- `scores::RealArray`: vector or matrix of pairwise scores.

# Description

Shares the same reductions between sample-to-cluster and cluster-to-cluster
comparisons. Weighted and centroid linkage require additional cluster state
and are handled by their specialized methods.
"""
function similarity(method::Symbol, scores::RealArray)
    method === :single && return single(scores)
    method === :complete && return complete(scores)
    method === :average && return average(scores)
    method === :median && return median(scores)
    throw(ArgumentError("Unsupported unweighted linkage method: $method"))
end

# -----------------------------------------------------------------------------
# DISTRIBUTED LINKAGE METHODS
# -----------------------------------------------------------------------------

# Argument docstring for the activation flag
const ACTIVATION_DOCSTRING = """
- `activation::Bool`: flag to use the activation function. False uses the match function.
"""

# Argument docstring for the sample vector
const SAMPLE_DOCSTRING = """
- `sample::RealVector`: the sample to use for computing the linkage to the F2 module.
"""

# Argument docstring for the F2 docstring
const F2_DOCSTRING = """
- `F2::FuzzyART`: the local FuzzyART F2 node to compute the linkage method within.
"""

"""
Compute the similarity metric depending on method with explicit comparisons for the field name.

# Arguments
- `method::Symbol`: the linkage method to use.
$F2_DOCSTRING
$SAMPLE_DOCSTRING
$ACTIVATION_DOCSTRING
"""
function similarity(method::Symbol, F2::FuzzyART, sample::RealVector, activation::Bool)
    # Centroid compares envelopes; other methods reduce the prepared score vector.
    method === :centroid && return centroid(F2, sample, activation)
    return similarity(method, activation ? F2.T : F2.M, F2)
end

"""
A list of similarity linkage methods supported by distributed ART models.
"""
const LINKAGE_METHODS = [
    :single,
    :average,
    :complete,
    :median,
    :weighted,
    :centroid,
]

"""
Weighted linkage distributed ART similarity function.

# Arguments:
$F2_DOCSTRING
$ACTIVATION_DOCSTRING
"""
function weighted(F2::FuzzyART, activation::Bool)
    # Select the prepared field before applying the shared weighted reduction.
    return weighted(activation ? F2.T : F2.M, F2)
end

"""
Centroid linkage distributed ART similarity function.

# Arguments:
$F2_DOCSTRING
$SAMPLE_DOCSTRING
$ACTIVATION_DOCSTRING
"""
function centroid(F2::FuzzyART, sample::RealVector, activation::Bool)
    Wc = cluster_envelope(F2)
    T = (norm(element_min(sample, Wc), 1) / (F2.opts.alpha + norm(Wc, 1))) ^ F2.opts.gamma

    if activation
        value = T
    else
        value = (norm(Wc, 1)^F2.opts.gamma_ref) * T
    end

    return value
end


"""
Compute the category probabilities of a local cluster.

# Arguments
- `art::FuzzyART`: nonempty cluster with positive instance counts.

# Description

Normalizes instance counts by their total for both sample-to-cluster and
cluster-to-cluster weighted linkage.
"""
category_probabilities(art::FuzzyART) = art.n_instance ./ sum(art.n_instance)

"""
Compute the fuzzy envelope of a local cluster.

# Arguments
- `art::FuzzyART`: nonempty cluster whose category weights form matrix columns.

# Description

Returns the componentwise minimum across prototypes. This envelope is the
centroid representation shared by distributed sample and cluster comparisons.
"""
cluster_envelope(art::FuzzyART) = vec(minimum(art.W, dims=2))

"""
Weight sample-to-category scores by the destination category probabilities.

# Arguments
- `scores::RealVector`: activation or match scores for destination categories.
- `destination::FuzzyART`: cluster supplying the corresponding instance counts.

# Description

Computes the expected score using the same probability helper as pairwise
cluster linkage. The supplied scores must correspond to the cluster categories.
"""
function weighted(scores::RealVector, destination::FuzzyART)
    return scores' * category_probabilities(destination)
end

"""
Weight pairwise scores by the category probabilities of both clusters.

# Arguments
- `scores::RealMatrix`: destination-by-input activation or match matrix.
- `destination::FuzzyART`: cluster supplying row instance counts.
- `input::FuzzyART`: cluster supplying column instance counts.

# Description

Normalizes each cluster's counts separately and weights each pair by the product
of its two category probabilities.
"""
function weighted(scores::RealMatrix, destination::FuzzyART, input::FuzzyART)
    p = category_probabilities(destination)
    q = category_probabilities(input)
    return sum(scores .* (p * q'))
end

"""
Reduce scores with the cluster context required by weighted linkage.

# Arguments
- `method::Symbol`: an unweighted linkage name or `:weighted`.
- `scores`: category scores as a vector or destination-by-input matrix.
- `destination::FuzzyART`: cluster supplying destination counts.
- `input::FuzzyART`: incoming cluster, required for pairwise matrix scores.

# Description

Weighted linkage uses one or both clusters' category probabilities. Other
methods delegate to the common unweighted reducer; centroid is evaluated from
cluster envelopes before score reduction, and unsupported names raise an error.
"""
function similarity(method::Symbol, scores::RealVector, destination::FuzzyART)
    method === :weighted && return weighted(scores, destination)
    return similarity(method, scores)
end

function similarity(method::Symbol, scores::RealMatrix, destination::FuzzyART, input::FuzzyART)
    method === :weighted && return weighted(scores, destination, input)
    return similarity(method, scores)
end
