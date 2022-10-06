"""
    DDVFA.jl

Description:
    Includes all of the structures and logic for running a Distributed Dual-Vigilance Fuzzy ART (DDVFA) module.

References
[1] L. E. Brito da Silva, I. Elnabarawy, and D. C. Wunsch, “Distributed dual vigilance fuzzy adaptive resonance theory learns online, retrieves arbitrarily-shaped clusters, and mitigates order dependence,” Neural Networks, vol. 121, pp. 208-228, 2020, doi: 10.1016/j.neunet.2019.08.033.
[2] G. Carpenter, S. Grossberg, and D. Rosen, "Fuzzy ART: Fast stable learning and categorization of analog patterns by an adaptive resonance system," Neural Networks, vol. 4, no. 6, pp. 759-771, 1991.
"""

# --------------------------------------------------------------------------- #
# OPTIONS
# --------------------------------------------------------------------------- #

"""
    opts_DDVFA(;kwargs)

Distributed Dual Vigilance Fuzzy ART options struct.

# Keyword Arguments
- `rho_lb::Float`: lower-bound vigilance value, [0, 1], default 0.7.
- `rho_ub::Float`: upper-bound vigilance value, [0, 1], default 0.85.
- `alpha::Float`: choice parameter, alpha > 0, default 1e-3.
- `beta::Float`: learning parameter, (0, 1], default 1.0.
- `gamma::Float`: "pseudo" kernel width, gamma >= 1, default 3.0.
- `gamma_ref::Float`: "reference" kernel width, 0 <= gamma_ref < gamma, default 1.0.
- `method::String`: similarity method (activation and match):
`single`, `average`, `complete`, `median`, `weighted`, or `centroid`, default `single`.
- `display::Bool`: display flag, default true.
- `max_epoch::Int`: maximum number of epochs during training, default 1.
- `gamma_normalization::Bool`: normalize the threshold by the feature dimension, default true.
"""
@with_kw mutable struct opts_DDVFA <: ARTOpts @deftype Float
    # Lower-bound vigilance parameter: [0, 1]
    rho_lb = 0.7; @assert rho_lb >= 0.0 && rho_lb <= 1.0
    # Upper bound vigilance parameter: [0, 1]
    rho_ub = 0.85; @assert rho_ub >= 0.0 && rho_ub <= 1.0
    # Choice parameter: alpha > 0
    alpha = 1e-3; @assert alpha > 0.0
    # Learning parameter: (0, 1]
    beta = 1.0; @assert beta > 0.0 && beta <= 1.0
    # "Pseudo" kernel width: gamma >= 1
    gamma = 3.0; @assert gamma >= 1.0
    # "Reference" gamma for normalization: 0 <= gamma_ref < gamma
    gamma_ref = 1.0; @assert 0.0 <= gamma_ref && gamma_ref < gamma
    # Similarity method (activation and match):
    #   'single', 'average', 'complete', 'median', 'weighted', or 'centroid'
    method::String = "single"
    # Display flag
    display::Bool = true
    # Maximum number of epochs during training
    max_epoch::Int = 1
    # Normalize the threshold by the feature dimension
    gamma_normalization::Bool = true
end # opts_DDVFA

# --------------------------------------------------------------------------- #
# STRUCTS
# --------------------------------------------------------------------------- #

"""
    DDVFA <: ART

Distributed Dual Vigilance Fuzzy ARTMAP module struct.

For module options, see [`AdaptiveResonance.opts_DDVFA`](@ref).

# Option Parameters
- `opts::opts_DDVFA`: DDVFA options struct.
- `subopts::opts_FuzzyART`: FuzzyART options struct used for all F2 nodes.
- `config::DataConfig`: data configuration struct.

# Working Parameters
- `threshold::Float`: operating module threshold value, a function of the vigilance parameter.
- `F2::Vector{FuzzyART}`: list of F2 nodes (themselves FuzzyART modules).
- `labels::Vector{Int}`: incremental list of labels corresponding to each F2 node, self-prescribed or supervised.
- `n_categories::Int`: number of total categories.
- `epoch::Int`: current training epoch.
- `T::Float`: winning activation value from most recent sample.
- `M::Float`: winning match value from most recent sample.

# References
1. L. E. Brito da Silva, I. Elnabarawy, and D. C. Wunsch, “Distributed dual vigilance fuzzy adaptive resonance theory learns online, retrieves arbitrarily-shaped clusters, and mitigates order dependence,” Neural Networks, vol. 121, pp. 208-228, 2020, doi: 10.1016/j.neunet.2019.08.033.
2. G. Carpenter, S. Grossberg, and D. Rosen, "Fuzzy ART: Fast stable learning and categorization of analog patterns by an adaptive resonance system," Neural Networks, vol. 4, no. 6, pp. 759-771, 1991.
"""
mutable struct DDVFA <: ART
    # Get parameters
    opts::opts_DDVFA
    subopts::opts_FuzzyART
    config::DataConfig

    # Working variables
    threshold::Float
    F2::Vector{FuzzyART}
    labels::Vector{Int}
    n_categories::Int
    epoch::Int
    T::Float
    M::Float
end # DDVFA <: ART

# --------------------------------------------------------------------------- #
# CONSTRUCTORS
# --------------------------------------------------------------------------- #

"""
    DDVFA(;kwargs...)

Implements a DDVFA learner with optional keyword arguments.

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
end # DDVFA(;kwargs...)

"""
    DDVFA(opts::opts_DDVFA)

Implements a DDVFA learner with specified options.

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
end # DDVFA(opts::opts_DDVFA)

# --------------------------------------------------------------------------- #
# ALGORITHMIC METHODS
# --------------------------------------------------------------------------- #

"""
    set_threshold!(art::DDVFA)

Sets the vigilance threshold of the DDVFA module as a function of several flags and hyperparameters.
"""
function set_threshold!(art::DDVFA)
    # Gamma match normalization
    if art.opts.gamma_normalization
        # Set the learning threshold as a function of the data dimension
        art.threshold = art.opts.rho_lb * (art.config.dim ^ art.opts.gamma_ref)
    else
        # Set the learning threshold as simply the vigilance parameter
        art.threshold = art.opts.rho_lb
    end
end # set_threshold!(art::DDVFA)

# DDVFA incremental training method
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
        T[jx] = similarity(art.opts.method, art.F2[jx], "T", sample, art.opts.gamma_ref)
    end

    # Compute the match for each category in the order of greatest activation
    index = sortperm(T, rev=true)
    for jx = 1:art.n_categories
        bmu = index[jx]
        M = similarity(art.opts.method, art.F2[bmu], "M", sample, art.opts.gamma_ref)
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
        art.M = similarity(art.opts.method, art.F2[bmu], "M", sample, art.opts.gamma_ref)
        art.T = T[bmu]
        # Get the correct label
        y_hat = supervised ? y : art.n_categories + 1
        create_category(art, sample, y_hat)
    end

    return y_hat
end # train!(art::DDVFA, x::RealVector ; y::Integer=0, preprocessed::Bool=false)

"""
    create_category(art::DDVFA, sample::RealVector, label::Integer)

Create a new category by appending and initializing a new FuzzyART node to F2.
"""
function create_category(art::DDVFA, sample::RealVector, label::Integer)
    # Global Fuzzy ART
    art.n_categories += 1
    push!(art.labels, label)
    # Local Gamma-Normalized Fuzzy ART
    push!(art.F2, FuzzyART(art.subopts, sample, preprocessed=true))
end # function create_category(art::DDVFA, sample::RealVector, label::Integer)

"""
    stopping_conditions(art::DDVFA)

Stopping conditions for Distributed Dual Vigilance Fuzzy ARTMAP.

Returns true if there is no change in weights during the epoch or the maxmimum epochs has been reached.
"""
function stopping_conditions(art::DDVFA)
    # Compute the stopping condition, return a bool
    return art.epoch >= art.opts.max_epoch
end # stopping_conditions(DDVFA)

"""
    similarity(method::String, F2::FuzzyART, field_name::String, sample::RealVector, gamma_ref::RealFP)

Compute the similarity metric depending on method with explicit comparisons
for the field name.
"""
function similarity(method::String, F2::FuzzyART, field_name::String, sample::RealVector, gamma_ref::RealFP)
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
            value = (norm(Wc, 1)^gamma_ref)*T
        end
    else
        error("Invalid/unimplemented similarity method")
    end

    return value
end # similarity(method::String, F2::FuzzyART, field_name::String, sample::RealVector, gamma_ref::RealFP)

# DDVFA incremental classification method
function classify(art::DDVFA, x::RealVector ; preprocessed::Bool=false, get_bmu::Bool=false)
    # Preprocess the data
    sample = init_classify!(x, art, preprocessed)

    # Calculate all global activations
    T = zeros(art.n_categories)
    for jx = 1:art.n_categories
        activation_match!(art.F2[jx], sample)
        T[jx] = similarity(art.opts.method, art.F2[jx], "T", sample, art.opts.gamma_ref)
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
        M = similarity(art.opts.method, art.F2[bmu], "M", sample, art.opts.gamma_ref)
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
        art.M = similarity(art.opts.method, art.F2[bmu], "M", sample, art.opts.gamma_ref)
        art.T = T[bmu]
        # Report either the best matching unit or the mismatch label -1
        y_hat = get_bmu ? art.labels[bmu] : -1
    end

    return y_hat
end

# --------------------------------------------------------------------------- #
# CONVENIENCE METHODS
# --------------------------------------------------------------------------- #

"""
    get_W(art::DDVFA)

Convenience functio; return a concatenated array of all DDVFA weights.
"""
function get_W(art::DDVFA)
    # Return a concatenated array of the weights
    return [art.F2[kx].W for kx = 1:art.n_categories]
end # get_W(art::DDVFA)

"""
    get_n_weights_vec(art::DDVFA)

Convenience function; return the number of weights in each category as a vector.
"""
function get_n_weights_vec(art::DDVFA)
    return [art.F2[i].n_categories for i = 1:art.n_categories]
end # get_n_weights_vec(art::DDVFA)

"""
    get_n_weights(art::DDVFA)

Convenience function; return the sum total number of weights in the DDVFA module.
"""
function get_n_weights(art::DDVFA)
    # Return the number of weights across all categories
    return sum(get_n_weights_vec(art))
end # get_n_weights(art::DDVFA)
