"""
    DDVFA.jl

# Description:
Includes all of the structures and logic for running a Distributed Dual-Vigilance Fuzzy ART (DDVFA) module.

# References
[1] L. E. Brito da Silva, I. Elnabarawy, and D. C. Wunsch, “Distributed dual vigilance fuzzy adaptive resonance theory learns online, retrieves arbitrarily-shaped clusters, and mitigates order dependence,” Neural Networks, vol. 121, pp. 208-228, 2020, doi: 10.1016/j.neunet.2019.08.033.
[2] G. Carpenter, S. Grossberg, and D. Rosen, "Fuzzy ART: Fast stable learning and categorization of analog patterns by an adaptive resonance system," Neural Networks, vol. 4, no. 6, pp. 759-771, 1991.
"""

# --------------------------------------------------------------------------- #
# OPTIONS
# --------------------------------------------------------------------------- #

"""
Distributed Dual Vigilance Fuzzy ART options struct.

$(opts_docstring)
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
    Similarity method (activation and match): method ∈ ["single", "average", "complete", "median", "weighted", "centroid"].
    """
    method::String = "single"

    """
    Maximum number of epochs during training: max_epochs ∈ (1, Inf).
    """
    max_epoch::Int = 1

    """
    Display flag.
    """
    display::Bool = true

    """
    Flag to normalize the threshold by the feature dimension.
    """
    gamma_normalization::Bool = true
end

# --------------------------------------------------------------------------- #
# STRUCTS
# --------------------------------------------------------------------------- #

"""
Distributed Dual Vigilance Fuzzy ARTMAP module struct.

For module options, see [`AdaptiveResonance.opts_DDVFA`](@ref).

# References
1. L. E. Brito da Silva, I. Elnabarawy, and D. C. Wunsch, “Distributed dual vigilance fuzzy adaptive resonance theory learns online, retrieves arbitrarily-shaped clusters, and mitigates order dependence,” Neural Networks, vol. 121, pp. 208-228, 2020, doi: 10.1016/j.neunet.2019.08.033.
2. G. Carpenter, S. Grossberg, and D. Rosen, "Fuzzy ART: Fast stable learning and categorization of analog patterns by an adaptive resonance system," Neural Networks, vol. 4, no. 6, pp. 759-771, 1991.
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
    labels::Vector{Int}

    """
    Number of total categories.
    """
    n_categories::Int

    """
    Current training epoch.
    """
    epoch::Int

    """
    Winning activation value from most recent sample.
    """
    T::Float

    """
    Winning match value from most recent sample.
    """
    M::Float
end

# --------------------------------------------------------------------------- #
# CONSTRUCTORS
# --------------------------------------------------------------------------- #

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
        display=false
    )

    # Construct the DDVFA module
    DDVFA(opts,
          subopts,
          DataConfig(),
          0.0,
          Array{FuzzyART}(undef, 0),
          Array{Int}(undef, 0),
          0,
          0,
          0.0,
          0.0
    )
end

# --------------------------------------------------------------------------- #
# COMMON FUNCTIONS
# --------------------------------------------------------------------------- #

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
        create_category(art, sample, y_hat)
        return y_hat
    end

    # Default to mismatch
    mismatch_flag = true

    # Compute the activation for all categories
    T = zeros(art.n_categories)
    for jx = 1:art.n_categories
        activation_match!(art.F2[jx], sample)
        T[jx] = similarity(art.opts.method, art.F2[jx], "T", sample)
    end

    # Compute the match for each category in the order of greatest activation
    index = sortperm(T, rev=true)
    for jx = 1:art.n_categories
        bmu = index[jx]
        M = similarity(art.opts.method, art.F2[bmu], "M", sample)
        # If we got a match, then learn (update the category)
        if M >= art.threshold
            # Update the stored match and activation values
            art.M = M
            art.T = T[bmu]
            # If supervised and the label differs, trigger mismatch
            if supervised && (art.labels[bmu] != y)
                break
            end
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
        art.M = similarity(art.opts.method, art.F2[bmu], "M", sample)
        art.T = T[bmu]
        # Get the correct label
        y_hat = supervised ? y : art.n_categories + 1
        create_category(art, sample, y_hat)
    end

    return y_hat
end

# COMMON DOC: DDVFA incremental classification method
function classify(art::DDVFA, x::RealVector ; preprocessed::Bool=false, get_bmu::Bool=false)
    # Preprocess the data
    sample = init_classify!(x, art, preprocessed)

    # Calculate all global activations
    T = zeros(art.n_categories)
    for jx = 1:art.n_categories
        activation_match!(art.F2[jx], sample)
        T[jx] = similarity(art.opts.method, art.F2[jx], "T", sample)
    end

    # Sort by highest activation
    index = sortperm(T, rev=true)

    # Default to mismatch
    mismatch_flag = true

    # Iterate over the list of activations
    for jx = 1:art.n_categories
        # Get the best-matching unit
        bmu = index[jx]
        # Get the match value of this activation
        M = similarity(art.opts.method, art.F2[bmu], "M", sample)
        # If the match satisfies the threshold criterion, then report that label
        if M >= art.threshold
            # Update the stored match and activation values
            art.M = M
            art.T = T[bmu]
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
        art.M = similarity(art.opts.method, art.F2[bmu], "M", sample)
        art.T = T[bmu]
        # Report either the best matching unit or the mismatch label -1
        y_hat = get_bmu ? art.labels[bmu] : -1
    end

    return y_hat
end

# --------------------------------------------------------------------------- #
# INTERNAL FUNCTIONS
# --------------------------------------------------------------------------- #

"""
Create a new category by appending and initializing a new FuzzyART node to F2.

# Arguments
- `art::DDVFA`: the DDVFA module to create a new FuzzyART category in.
- `sample::RealVector`: the sample to use for instantiating the new category.
- `label::Integer`: the new label to use for the new category.
"""
function create_category(art::DDVFA, sample::RealVector, label::Integer)
    # Global Fuzzy ART
    art.n_categories += 1
    push!(art.labels, label)
    # Local Gamma-Normalized Fuzzy ART
    push!(art.F2, FuzzyART(art.subopts, sample, preprocessed=true))
end

"""
Stopping conditions for Distributed Dual Vigilance Fuzzy ARTMAP.

Returns true if there is no change in weights during the epoch or the maxmimum epochs has been reached.

# Arguments
- `art::DDVFA`: the DDVFA module for checking stopping conditions.
"""
function stopping_conditions(art::DDVFA)
    # Compute the stopping condition, return a bool
    return art.epoch >= art.opts.max_epoch
end

"""
Compute the similarity metric depending on method with explicit comparisons for the field name.

# Arguments
- `method::AbstractString`: the selected DDVFA linkage method.
- `F2::FuzzyART`: the FuzzyART module to compute the linkage method within.
- `field_name::AbstractString`: the activation or match value to compute, field_name ∈ ["T", "M"]
- `sample::RealVector`: the sample to use for computing the linkage to the F2 module, sample ∈ DDVFA_METHODS.
"""
function similarity(method::AbstractString, F2::FuzzyART, field_name::AbstractString, sample::RealVector)
    @debug "Computing similarity"

    if field_name != "T" && field_name != "M"
        error("Incorrect field name for similarity metric.")
    end
    # Single linkage
    if method == "single"
        if field_name == "T"
            value = maximum(F2.T)
        elseif field_name == "M"
            value = maximum(F2.M)
        end
    # Average linkage
    elseif method == "average"
        if field_name == "T"
            value = mean(F2.T)
        elseif field_name == "M"
            value = mean(F2.M)
        end
    # Complete linkage
    elseif method == "complete"
        if field_name == "T"
            value = minimum(F2.T)
        elseif field_name == "M"
            value = minimum(F2.M)
        end
    # Median linkage
    elseif method == "median"
        if field_name == "T"
            value = median(F2.T)
        elseif field_name == "M"
            value = median(F2.M)
        end
    # Weighted linkage
    elseif method == "weighted"
        if field_name == "T"
            value = F2.T' * (F2.n_instance ./ sum(F2.n_instance))
        elseif field_name == "M"
            value = F2.M' * (F2.n_instance ./ sum(F2.n_instance))
        end
    # Centroid linkage
    elseif method == "centroid"
        # Get the minimum of each weight element, cast to a 1-D vector
        Wc = vec(minimum(F2.W, dims=2))
        T = norm(element_min(sample, Wc), 1) / (F2.opts.alpha + norm(Wc, 1))^F2.opts.gamma
        if field_name == "T"
            value = T
        elseif field_name == "M"
            value = (norm(Wc, 1)^F2.opts.gamma_ref)*T
        end
    else
        error("Invalid/unimplemented similarity method")
    end

    return value
end

# --------------------------------------------------------------------------- #
# CONVENIENCE METHODS
# --------------------------------------------------------------------------- #

"""
Convenience function; return a concatenated array of all DDVFA weights.
"""
function get_W(art::DDVFA)
    # Return a concatenated array of the weights
    return [art.F2[kx].W for kx = 1:art.n_categories]
end

"""
Convenience function; return the number of weights in each category as a vector.
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
