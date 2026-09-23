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
