"""
    symbols.jl

# Description
Symbols for macro evaluation of activation, match, and learning functions.
"""

# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------

"""
Low-level common function for computing the 1-norm of the element minimum of a sample and weights.

# Arguments
$(_ARG_X)
$(_ARG_W)
"""
function x_W_min_norm(x::RealVector, W::RealVector)
    # return @inbounds norm(element_min(x, get_sample(W, index)), 1)
    return norm(element_min(x, W), 1)
end

"""
Low-level common function for computing the 1-norm of just the weight vector.

# Arguments
$(_ARG_W)
"""
function W_norm(W::RealVector)
    return norm(W, 1)
end

"""
Basic match function.

$(_ARG_ART_X_W)
"""
function basic_match(art::ARTModule, x::RealVector, W::RealVector)
    # return norm(element_min(x, get_sample(W, index)), 1) / art.config.dim
    return x_W_min_norm(x, W) / art.config.dim
end

"""
Unnormalized match function.

$(_ARG_ART_X_W)
"""
function unnormalized_match(_::ARTModule, x::RealVector, W::RealVector)
    # return norm(element_min(x, get_sample(W, index)), 1) / art.config.dim
    return x_W_min_norm(x, W)
end

"""
Simplified FuzzyARTMAP activation function.

$(_ARG_ART_X_W)
"""
function basic_activation(art::ARTModule, x::RealVector, W::RealVector)
    # return norm(element_min(x, get_sample(W, index)), 1) / (art.opts.alpha + norm(get_sample(W, index), 1))
    return x_W_min_norm(x, W) / (art.opts.alpha + W_norm(W))
end

"""
Low-level subroutine for the gamma match function with a precomputed gamma activation.

# Arguments
$(_ARG_ART)
$(_ARG_W)
- `gamma_act::Real`: the precomputed gamma activation value.
"""
function gamma_match_sub(art::ARTModule, W::RealVector, gamma_act::Real)
    return (W_norm(W) ^ art.opts.gamma_ref) * gamma_act
end

"""
Gamma-normalized match function, recomputing the gamma activation value.

$(_ARG_ART_X_W)
"""
function gamma_match(art::ARTModule, x::RealVector, W::RealVector)
    return gamma_match_sub(art, W, gamma_activation(art, x, W))
end

"""
Gamma-normalized match function, passing a precomputed gamma activation value.

$(_ARG_ART_X_W)
- `gamma_act::Real`: the precomputed gamma activation value.
"""
function gamma_match(art::ARTModule, _::RealVector, W::RealVector, gamma_act::Real)
    return gamma_match_sub(art, W, gamma_act::Real)
end

"""
Gamma-normalized activation funtion.

$(_ARG_ART_X_W)
"""
function gamma_activation(art::ARTModule, x::RealVector, W::RealVector)
    return basic_activation(art, x, W) ^ art.opts.gamma
end

"""
Default ARTMAP's choice-by-difference activation function.

$(_ARG_ART_X_W)
"""
function choice_by_difference(art::ARTModule, x::RealVector, W::RealVector)
    return (
        x_W_min_norm(x, W)
            + (1 - art.opts.alpha) * (art.config.dim - W_norm(W))
    )
end

"""
Evaluates the match function of the ART/ARTMAP module on sample 'x' with weight 'W'.

Passes additional arguments for low-level optimizations using function dispatch.

# Arguments
$(_ARG_ART)
$(_ARG_X)
$(_ARG_INDEX)
"""
function art_match(art::ARTModule, x::RealVector, index::Integer, args...)
    return eval(art.opts.match)(art, x, get_sample(art.W, index), args...)
end

"""
Evaluates the activation function of the ART/ARTMAP module on the sample 'x' with weight 'W'.

Passes additional arguments for low-level optimizations using function dispatch.

# Arguments
$(_ARG_ART)
$(_ARG_X)
$(_ARG_INDEX)
"""
function art_activation(art::ARTModule, x::RealVector, index::Integer, args...)
    return eval(art.opts.activation)(art, x, get_sample(art.W, index), args...)
end

"""
Basic weight update function.

$(_ARG_ART_X_W)
"""
function basic_update(art::ARTModule, x::RealVector, W::RealVector)
    return art.opts.beta * element_min(x, W) + W * (1.0 - art.opts.beta)
end

"""
Evaluates the ART module's learning/update method.

# Arguments
$(_ARG_ART)
$(_ARG_X)
$(_ARG_INDEX)
"""
function art_learn(art::ARTModule, x::RealVector, index::Integer)
    return eval(art.opts.update)(art, x, get_sample(art.W, index))
end

# -----------------------------------------------------------------------------
# ENUMERATIONS
# -----------------------------------------------------------------------------

"""
Enumerates all of the update functions available in the package.
"""
const UPDATE_FUNCTIONS = [
    :basic_update,
]

"""
Enumerates all of the match functions available in the package.
"""
const MATCH_FUNCTIONS = [
    :basic_match,
    :gamma_match,
]

"""
Enumerates all of the activation functions available in the package.
"""
const ACTIVATION_FUNCTIONS = [
    :basic_activation,
    :unnormalized_match,
    :choice_by_difference,
    :gamma_activation,
]
