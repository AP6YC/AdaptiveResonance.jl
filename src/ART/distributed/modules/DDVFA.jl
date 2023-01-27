"""
    DDVFA.jl

# Description
Includes all of the structures and logic for running a Distributed Dual-Vigilance Fuzzy ART (DDVFA) module.

# References
1. L. E. Brito da Silva, I. Elnabarawy, and D. C. Wunsch, 'Distributed dual vigilance fuzzy adaptive resonance theory learns online, retrieves arbitrarily-shaped clusters, and mitigates order dependence,' Neural Networks, vol. 121, pp. 208-228, 2020, doi: 10.1016/j.neunet.2019.08.033.
2. G. Carpenter, S. Grossberg, and D. Rosen, 'Fuzzy ART: Fast stable learning and categorization of analog patterns by an adaptive resonance system,' Neural Networks, vol. 4, no. 6, pp. 759-771, 1991.
"""

# -----------------------------------------------------------------------------
# OPTIONS
# -----------------------------------------------------------------------------

"""
Distributed Dual Vigilance Fuzzy ART options struct.

$(OPTS_DOCSTRING)
"""
@with_kw mutable struct opts_DDVFA <: ARTOpts @deftype Float
    """
    Lower-bound vigilance parameter: rho_lb ∈ [0, 1].
    """
    rho_lb = 0.7; @assert rho_lb >= 0.0 && rho_lb <= 1.0

    """
    Upper bound vigilance parameter: rho_ub ∈ [0, 1].
    """
    rho_ub = 0.85; @assert rho_ub >= 0.0 && rho_ub <= 1.0

    """
    Choice parameter: alpha > 0.
    """
    alpha = 1e-3; @assert alpha > 0.0

    """
    Learning parameter: beta ∈ (0, 1].
    """
    beta = 1.0; @assert beta > 0.0 && beta <= 1.0

    """
    Pseudo kernel width: gamma >= 1.
    """
    gamma = 3.0; @assert gamma >= 1.0

    """
    Reference gamma for normalization: 0 <= gamma_ref < gamma.
    """
    gamma_ref = 1.0; @assert 0.0 <= gamma_ref && gamma_ref < gamma

    """
    Similarity method (activation and match): similarity ∈ [:single, :average, :complete, :median, :weighted, :centroid].
    """
    similarity::Symbol = :single

    """
    Maximum number of epochs during training: max_epochs ∈ (1, Inf).
    """
    max_epoch::Int = 1

    """
    Display flag for progress bars.
    """
    display::Bool = false

    """
    Flag to normalize the threshold by the feature dimension.
    """
    gamma_normalization::Bool = true

    """
    Flag to use an uncommitted node when learning.

    If true, new weights are created with ones(dim) and learn on the complement-coded sample.
    If false, fast-committing is used where the new weight is simply the complement-coded sample.
    """
    uncommitted::Bool = false

    """
    Selected activation function.
    """
    activation::Symbol = :gamma_activation

    """
    Selected match function.
    """
    match::Symbol = :gamma_match
end

# -----------------------------------------------------------------------------
# STRUCTS
# -----------------------------------------------------------------------------

"""
Distributed Dual Vigilance Fuzzy ARTMAP module struct.

For module options, see [`AdaptiveResonance.opts_DDVFA`](@ref).

# References
1. L. E. Brito da Silva, I. Elnabarawy, and D. C. Wunsch, 'Distributed dual vigilance fuzzy adaptive resonance theory learns online, retrieves arbitrarily-shaped clusters, and mitigates order dependence,' Neural Networks, vol. 121, pp. 208-228, 2020, doi: 10.1016/j.neunet.2019.08.033.
2. G. Carpenter, S. Grossberg, and D. Rosen, 'Fuzzy ART: Fast stable learning and categorization of analog patterns by an adaptive resonance system,' Neural Networks, vol. 4, no. 6, pp. 759-771, 1991.
"""
mutable struct DDVFA <: ART
    # Option Parameters
    """
    DDVFA options struct.
    """
    opts::opts_DDVFA

    """
    FuzzyART options struct used for all F2 nodes.
    """
    subopts::opts_FuzzyART

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
    DDVFA activation values.
    """
    T::ARTVector

    """
    DDVFA match values.
    """
    M::ARTVector

    """
    Winning activation value from most recent sample.
    """
    T_win::Float

    """
    Winning match value from most recent sample.
    """
    M_win::Float
end

# -----------------------------------------------------------------------------
# CONSTRUCTORS
# -----------------------------------------------------------------------------

"""
Implements a DDVFA learner with optional keyword arguments.

# Arguments
- `kwargs`: keyword arguments to pass to the DDVFA options struct (see [`AdaptiveResonance.opts_DDVFA`](@ref).)

# Examples
By default:
```julia-repl
julia> DDVFA()
DDVFA
    opts: opts_DDVFA
    subopts: opts_FuzzyART
    ...
```

or with keyword arguments:
```julia-repl
julia> DDVFA(rho_lb=0.4, rho_ub = 0.75)
DDVFA
    opts: opts_DDVFA
    subopts: opts_FuzzyART
    ...
```
"""
function DDVFA(;kwargs...)
    opts = opts_DDVFA(;kwargs...)
    DDVFA(opts)
end

"""
Implements a DDVFA learner with specified options.

# Arguments
- `opts::opts_DDVFA`: the DDVFA options (see [`AdaptiveResonance.opts_DDVFA`](@ref)).

# Examples
```julia-repl
julia> my_opts = opts_DDVFA()
julia> DDVFA(my_opts)
DDVFA
    opts: opts_DDVFA
    subopts: opts_FuzzyART
    ...
```
"""
function DDVFA(opts::opts_DDVFA)
    # Set the options used for all F2 FuzzyART modules
    subopts = opts_FuzzyART(
        rho=opts.rho_ub,
        gamma=opts.gamma,
        gamma_ref=opts.gamma_ref,
        gamma_normalization=opts.gamma_normalization,
        uncommitted=opts.uncommitted,
        display=false,
        activation=opts.activation,
        match=opts.match
    )

    # Construct the DDVFA module
    DDVFA(opts,
          subopts,
          DataConfig(),
          0.0,
          Vector{FuzzyART}(undef, 0),
          ARTVector{Int}(undef, 0),
          0,
          0,
          ARTVector{Float}(undef, 0),
          ARTVector{Float}(undef, 0),
          0.0,
          0.0
    )
end

# -----------------------------------------------------------------------------
# COMMON FUNCTIONS
# -----------------------------------------------------------------------------

# COMMON DOC: Set threshold function
function set_threshold!(art::DDVFA)
    # Gamma match normalization
    if art.opts.gamma_normalization
        # Set the learning threshold as a function of the data dimension
        art.threshold = art.opts.rho_lb * (art.config.dim ^ art.opts.gamma_ref)
    else
        # Set the learning threshold as simply the vigilance parameter
        art.threshold = art.opts.rho_lb
    end
end

# COMMON DOC: DDVFA incremental training method
function train!(art::DDVFA, x::RealVector ; y::Integer=0, preprocessed::Bool=false)
    # Flag for if training in supervised mode
    supervised = !iszero(y)

    # Run the sequential initialization procedure
    sample = init_train!(x, art, preprocessed)

    # Initialization
    if isempty(art.F2)
        # Set the threshold
        set_threshold!(art)
        # Set the first label as either 1 or the first provided label
        y_hat = supervised ? y : 1
        # Create a new category
        create_category!(art, sample, y_hat)
        return y_hat
    end

    # Default to mismatch
    mismatch_flag = true

    # Compute the activation for all categories
    # T = zeros(art.n_categories)
    accommodate_vector!(art.T, art.n_categories)
    accommodate_vector!(art.M, art.n_categories)
    for jx = 1:art.n_categories
        activation_match!(art.F2[jx], sample)
        art.T[jx] = similarity(art.opts.similarity, art.F2[jx], sample, true)
    end

    # Compute the match for each category in the order of greatest activation
    index = sortperm(art.T, rev=true)
    for jx = 1:art.n_categories
        bmu = index[jx]
        # If supervised and the label differs, trigger mismatch
        if supervised && (art.labels[bmu] != y)
            break
        end

        # M = similarity(art.opts.similarity, art.F2[jx], sample, false)
        # art.M[jx] = similarity(art.opts.similarity, art.F2[jx], sample, false)
        art.M[bmu] = similarity(art.opts.similarity, art.F2[bmu], sample, false)
        # If we got a match, then learn (update the category)
        # if M >= art.threshold
        if art.M[bmu] >= art.threshold
            # Update the stored match and activation values
            # art.M_win = M
            # art.T_win = T[bmu]
            art.M_win = art.M[bmu]
            art.T_win = art.T[bmu]
            # Update the weights with the sample
            train!(art.F2[bmu], sample, preprocessed=true)
            # Save the output label for the sample
            y_hat = art.labels[bmu]
            # No mismatch
            mismatch_flag = false
            break
        end
    end

    # If we triggered a mismatch
    if mismatch_flag
        # Update the stored match and activation values
        bmu = index[1]
        # art.M = similarity(art.opts.similarity, art.F2[bmu], sample, false)
        # art.T_win = T[bmu]
        art.M_win = similarity(art.opts.similarity, art.F2[bmu], sample, false)
        art.T_win = art.T[bmu]
        # Get the correct label
        y_hat = supervised ? y : art.n_categories + 1
        create_category!(art, sample, y_hat)
    end

    return y_hat
end

# COMMON DOC: DDVFA incremental classification method
function classify(art::DDVFA, x::RealVector ; preprocessed::Bool=false, get_bmu::Bool=false)
    # Preprocess the data
    sample = init_classify!(x, art, preprocessed)

    # Calculate all global activations
    # T = zeros(art.n_categories)
    accommodate_vector!(art.T, art.n_categories)
    accommodate_vector!(art.M, art.n_categories)
    for jx = 1:art.n_categories
        activation_match!(art.F2[jx], sample)
        # T[jx] = similarity(art.opts.similarity, art.F2[jx], sample, true)
        art.T[jx] = similarity(art.opts.similarity, art.F2[jx], sample, true)
    end

    # Sort by highest activation
    # index = sortperm(T, rev=true)
    index = sortperm(art.T, rev=true)

    # Default to mismatch
    mismatch_flag = true

    # Iterate over the list of activations
    for jx = 1:art.n_categories
        # Get the best-matching unit
        bmu = index[jx]
        # Get the match value of this activation
        # M = similarity(art.opts.similarity, art.F2[bmu], sample, false)
        # art.M[jx] = similarity(art.opts.similarity, art.F2[bmu], sample, false)
        art.M[bmu] = similarity(art.opts.similarity, art.F2[bmu], sample, false)
        # If the match satisfies the threshold criterion, then report that label
        # if M >= art.threshold
        if art.M[bmu] >= art.threshold
            # Update the stored match and activation values
            # art.M_win = M
            # art.T_win = T[bmu]
            art.M_win = art.M[bmu]
            art.T_win = art.T[bmu]
            # Current winner
            y_hat = art.labels[bmu]
            mismatch_flag = false
            break
        end
    end

    # If we did not find a resonant category
    if mismatch_flag
        @debug "Mismatch"
        # Update the stored match and activation values of the best matching unit
        bmu = index[1]
        art.M_win = similarity(art.opts.similarity, art.F2[bmu], sample, false)
        art.T_win = art.T[bmu]
        # Report either the best matching unit or the mismatch label -1
        y_hat = get_bmu ? art.labels[bmu] : -1
    end

    return y_hat
end

# -----------------------------------------------------------------------------
# INTERNAL FUNCTIONS
# -----------------------------------------------------------------------------

"""
Create a new category by appending and initializing a new FuzzyART node to F2.

# Arguments
- `art::DDVFA`: the DDVFA module to create a new FuzzyART category in.
- `sample::RealVector`: the sample to use for instantiating the new category.
- `label::Integer`: the new label to use for the new category.
"""
function create_category!(art::DDVFA, sample::RealVector, label::Integer)
    # Global Fuzzy ART
    art.n_categories += 1
    push!(art.labels, label)
    # Local Gamma-Normalized Fuzzy ART
    push!(art.F2, FuzzyART(art.subopts, sample, preprocessed=true))
end

# -----------------------------------------------------------------------------
# DDVFA LINKAGE METHODS
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
- `F2::FuzzyART`: the DDVFA FuzzyART F2 node to compute the linkage method within.
"""

# Argument docstring for the F2 field, includes the argument header
const FIELD_DOCSTRING = """
# Arguments
- `field::RealVector`: the DDVFA FuzzyART F2 node field (F2.T or F2.M) to compute the linkage for.
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
    # Handle :centroid usage
    if method === :centroid
        value = eval(method)(F2, sample, activation)
    # Handle :weighted usage
    elseif method === :weighted
        value = eval(method)(F2, activation)
    # Handle common usage
    else
        value = eval(method)(activation ? F2.T : F2.M)
    end

    return value
end

"""
A list of all DDVFA similarity linkage methods.
"""
const DDVFA_METHODS = [
    :single,
    :average,
    :complete,
    :median,
    :weighted,
    :centroid,
]

"""
Single linkage DDVFA similarity function.

$FIELD_DOCSTRING
"""
function single(field::RealVector)
    return maximum(field)
end

"""
Average linkage DDVFA similarity function.

$FIELD_DOCSTRING
"""
function average(field::RealVector)
    return statistics_mean(field)
end

"""
Complete linkage DDVFA similarity function.

$FIELD_DOCSTRING
"""
function complete(field::RealVector)
    return minimum(field)
end

"""
Median linkage DDVFA similarity function.

$FIELD_DOCSTRING
"""
function median(field::RealVector)
    return statistics_median(field)
end

"""
Weighted linkage DDVFA similarity function.

# Arguments:
$F2_DOCSTRING
$ACTIVATION_DOCSTRING
"""
function weighted(F2::FuzzyART, activation::Bool)
    if activation
        value = F2.T' * (F2.n_instance ./ sum(F2.n_instance))
    else
        value = F2.M' * (F2.n_instance ./ sum(F2.n_instance))
    end

    return value
end

"""
Centroid linkage DDVFA similarity function.

# Arguments:
$F2_DOCSTRING
$SAMPLE_DOCSTRING
$ACTIVATION_DOCSTRING
"""
function centroid(F2::FuzzyART, sample::RealVector, activation::Bool)
    Wc = vec(minimum(F2.W, dims=2))
    T = norm(element_min(sample, Wc), 1) / (F2.opts.alpha + norm(Wc, 1))^F2.opts.gamma

    if activation
        value = T
    else
        value = (norm(Wc, 1)^F2.opts.gamma_ref) * T
    end

    return value
end

# -----------------------------------------------------------------------------
# CONVENIENCE METHODS
# -----------------------------------------------------------------------------

"""
Convenience function; return a concatenated array of all DDVFA weights.

# Arguments
- `art::DDVFA`: the DDVFA module to get all of the weights from as a list.
"""
function get_W(art::DDVFA)
    # Return a concatenated array of the weights
    return [art.F2[kx].W for kx = 1:art.n_categories]
end

"""
Convenience function; return the number of weights in each category as a vector.

# Arguments
- `art::DDVFA`: the DDVFA module to get all of the weights from as a list.
"""
function get_n_weights_vec(art::DDVFA)
    return [art.F2[i].n_categories for i = 1:art.n_categories]
end

"""
Convenience function; return the sum total number of weights in the DDVFA module.
"""
function get_n_weights(art::DDVFA)
    # Return the number of weights across all categories
    return sum(get_n_weights_vec(art))
end
