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

# COMMON DOC: FuzzyART initialization function
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

# """
# Initializes a DVFA learner with an initial sample 'x'.

# This function is used during the first training iteraction when the DVFA module is empty.

# # Arguments
# - `art::DVFA`: the DVFA module to initialize.
# - `x::RealVector`: the sample to use for initialization.
# - `y::Integer=0`: the optional new label for the first weight of the FuzzyART module. If not specified, defaults the new label to 1.
# """
# function initialize!(art::DVFA, x::RealVector ; y::Integer=0)
#     # Set the threshold
#     set_threshold!(art)
#     # Initialize the empty weight matrix to the correct dimension
#     art.W = ARTMatrix{Float}(undef, art.config.dim_comp, 0)
#     # Set the label to either the supervised label or 1 if unsupervised
#     label = !iszero(y) ? y : 1
#     # Create a new category
#     create_category!(art, x, label)
# end

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

# """
# Compute and store the activation and match values for the DVFA module.
# """
# function activation_match!(art::DVFA, x::RealVector)
#     # Expand the destination activation and match vectors
#     accommodate_vector!(art.T, art.n_categories)
#     accommodate_vector!(art.M, art.n_categories)
#     for jx = 1:art.n_categories
#         art.T[jx] = art_activation(art, x, jx)
#         art.M[jx] = art_match(art, x, jx)
#     end
# end

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
    # # Increment the instance counting
    # art.n_instance[index] += 1
    # Return empty
    return
end

"""
Stopping conditions for a FuzzyART module.

# Arguments
- `art::AbstractFuzzyART`: the FuzzyART module to check stopping conditions for.
"""
function stopping_conditions(art::AbstractFuzzyART)
    return art.epoch >= art.opts.max_epochs
end
