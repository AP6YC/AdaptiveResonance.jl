"""
    common.jl

# Description
Contains all common code for single ART modules (i.e. not distributed models).
"""

# -----------------------------------------------------------------------------
# TYPES
# -----------------------------------------------------------------------------

"""
Abstract supertype of FuzzyART modules.
"""
abstract type AbstractFuzzyART <: ART end

# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------

# COMMON DOC: AbstractFuzzyART initialization function
function initialize!(art::AbstractFuzzyART, x::RealVector ; y::Integer=0)
    # Set the threshold
    set_threshold!(art)
    # Initialize the feature dimension of the weights
    art.W = ARTMatrix{Float}(undef, art.config.dim_comp, 0)
    # Set the label to either the supervised label or 1 if unsupervised
    label = !iszero(y) ? y : 1
    # Create a category with the given label
    create_category!(art, x, label)
end

"""
Computes the activation and match functions of the ART module against sample x.

# Arguments
- `art::AbstractFuzzyART`: the single FuzzyART module to compute the activation and match values for all weights.
- `x::RealVector`: the sample to compute the activation and match functions against.

# Examples
```julia-repl
julia> my_FuzzyART = FuzzyART()
FuzzyART
    opts: opts_FuzzyART
    ...
julia> x = rand(3, 10)
julia> train!(my_FuzzyART, x)
julia> activation_match!(my_FuzzyART, x[:, 1])
```
"""
function activation_match!(art::AbstractFuzzyART, x::RealVector)
    # Expand the destination activation and match vectors
    accommodate_vector!(art.T, art.n_categories)
    accommodate_vector!(art.M, art.n_categories)
    for i = 1:art.n_categories
        art.T[i] = art_activation(art, x, i)
        # If we are using gamma normalization, save some computation
        if (art.opts.match == :gamma_match) && (art.opts.activation == :gamma_activation)
            art.M[i] = art_match(art, x, i, art.T[i])
        else
            art.M[i] = art_match(art, x, i)
        end
    end
end

"""
In place learning function.

# Arguments
- `art::AbstractFuzzyART`: the FuzzyART module to update.
- `x::RealVector`: the sample to learn from.
- `index::Integer`: the index of the FuzzyART weight to update.
"""
function learn!(art::AbstractFuzzyART, x::RealVector, index::Integer)
    # Compute the updated weight W
    new_vec = art_learn(art, x, index)
    # Replace the weight in place
    replace_mat_index!(art.W, new_vec, index)
    # Return empty
    return
end
