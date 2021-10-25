"""
    DDVFA.jl

Description:
    Includes all of the structures and logic for running a Distributed Dual-Vigilance Fuzzy ART (DDVFA) module.
"""

# --------------------------------------------------------------------------- #
# OPTIONS
# --------------------------------------------------------------------------- #

"""
    opts_DDVFA()

Distributed Dual Vigilance Fuzzy ART options struct.

# Examples
```julia-repl
julia> my_opts = opts_DDVFA()
```
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

# Examples
```julia-repl
julia> DDVFA()
DDVFA
    opts: opts_DDVFA
    subopts::opts_FuzzyART
    ...
```
"""
mutable struct DDVFA <: ART
    # Get parameters
    opts::opts_DDVFA
    subopts::opts_FuzzyART
    config::DataConfig

    # Working variables
    threshold::Float
    F2::Vector{FuzzyART}
    labels::IntegerVector
    n_categories::Int
    epoch::Int
    T::Float
    M::Float
end # DDVFA <: ART

# --------------------------------------------------------------------------- #
# CONSTRUCTORS
# --------------------------------------------------------------------------- #

"""
    DDVFA()

Implements a DDVFA learner with default options.

# Examples
```julia-repl
julia> DDVFA()
DDVFA
    opts: opts_DDVFA
    subopts: opts_FuzzyART
    ...
```
"""
function DDVFA()
    opts = opts_DDVFA()
    DDVFA(opts)
end # DDVFA()

"""
    DDVFA(;kwargs...)

Implements a DDVFA learner with keyword arguments.

# Examples
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
        art.threshold = art.opts.rho_lb*(art.config.dim^art.opts.gamma_ref)
    else
        # Set the learning threshold as simply the vigilance parameter
        art.threshold = art.opts.rho_lb
    end
end # set_threshold!(art::DDVFA)

"""
    train!(art::DDVFA, x::RealVector ; y::Integer=0, preprocessed::Bool=false)
"""
function train!(art::DDVFA, x::RealVector ; y::Integer=0, preprocessed::Bool=false)
    # Flag for if training in supervised mode
    supervised = !iszero(y)

    # Run the sequential initialization procedure
    sample = init_train!(x, art, preprocessed)

    # Initialization
    if isempty(art.F2)
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
    train!(art::DDVFA, x::RealMatrix ; y::IntegerVector=Vector{Int}(), preprocessed::Bool=false)

Train the DDVFA model on the data.
"""
function train!(art::DDVFA, x::RealMatrix ; y::IntegerVector = Vector{Int}(), preprocessed::Bool=false)
    # Show a message if display is on
    art.opts.display && @info "Training DDVFA"

    # Flag for if training in supervised mode
    supervised = !isempty(y)

    # Data information and setup
    n_samples = get_n_samples(x)

    # Run the batch initialization procedure
    x = init_train!(x, art, preprocessed)

    # Set the learning threshold
    set_threshold!(art)

    # Initialize the output vector
    y_hat = zeros(Int, n_samples)
    # Learn until the stopping conditions
    art.epoch = 0
    while true
        # Increment the epoch and get the iterator
        art.epoch += 1
        iter = get_iterator(art.opts, x)
        for i = iter
            # Update the iterator if necessary
            update_iter(art, iter, i)
            # Grab the sample slice
            sample = get_sample(x, i)
            # Select the label to pass to the incremental method
            local_y = supervised ? y[i] : 0
            # Train upon the sample and label
            y_hat[i] = train!(art, sample, y=local_y, preprocessed=true)
        end

        # Check stopping conditions
        if stopping_conditions(art)
            break
        end
    end
    return y_hat
end # train!(art::DDVFA, x::RealMatrix ; y::IntegerVector = Vector{Int}(), preprocessed::Bool=false)

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
        # If falling back to the highest activated category, return that
        if get_bmu
            y_hat = art.labels[bmu]
        # Otherwise, return a mismatch
        else
            y_hat = -1
        end
    end

    return y_hat
end

"""
    classify(art::DDVFA, x::RealMatrix ; preprocessed::Bool=false, get_bmu::Bool=false)

Predict categories of 'x' using the DDVFA model.

Returns predicted categories 'y_hat.'

# Examples
```julia-repl
julia> my_DDVFA = DDVFA()
DDVFA
    opts: opts_DDVFA
    ...
julia> x, y = load_data()
julia> train!(my_DDVFA, x)
julia> y_hat = classify(my_DDVFA, y)
```
"""
function classify(art::DDVFA, x::RealMatrix ; preprocessed::Bool=false, get_bmu::Bool=false)
    # Show a message if display is on
    art.opts.display && @info "Testing DDVFA"

    # Preprocess the data
    x = init_classify!(x, art, preprocessed)

    # Data information and setup
    n_samples = get_n_samples(x)

    # Initialize the output vector
    y_hat = zeros(Int, n_samples)

    # Get the iterator based on the module options and data shape
    iter = get_iterator(art.opts, x)
    for ix = iter
        # Update the iterator if necessary
        update_iter(art, iter, ix)

        # Grab the sample slice
        sample = get_sample(x, ix)

        # Get the classification
        y_hat[ix] = classify(art, sample, preprocessed=true, get_bmu=get_bmu)
    end

    return y_hat
end # classify(art::DDVFA, x::RealMatrix ; preprocessed::Bool=false, get_bmu::Bool=false)

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
