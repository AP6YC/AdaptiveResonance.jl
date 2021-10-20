"""
    DDVFA.jl

Description:
    Includes all of the structures and logic for running a Distributed Dual-Vigilance Fuzzy ART (DDVFA) module.
"""

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
    rho_lb = 0.80; @assert rho_lb >= 0.0 && rho_lb <= 1.0
    rho = rho_lb
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
end # opts_DDVFA

"""
    DDVFA <: ART

Distributed Dual Vigilance Fuzzy ARTMAP module struct.

# Examples
```julia-repl
julia> DDVFA()
DDVFA
    opts: opts_DDVFA
    subopts::opts_GNFA
    ...
```
"""
mutable struct DDVFA <: ART
    # Get parameters
    opts::opts_DDVFA
    subopts::opts_GNFA
    config::DataConfig

    # Working variables
    threshold::Float
    F2::Vector{GNFA}
    labels::IntegerVector
    W::RealMatrix        # All F2 nodes' weight vectors
    W_old::RealMatrix    # Old F2 node weight vectors (for stopping criterion)
    n_categories::Int
    epoch::Int
    T::Float
    M::Float
end # DDVFA <: ART

"""
    DDVFA()

Implements a DDVFA learner with default options.

# Examples
```julia-repl
julia> DDVFA()
DDVFA
    opts: opts_DDVFA
    subopts: opts_GNFA
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
julia> DDVFA(rho=0.7)
DDVFA
    opts: opts_DDVFA
    subopts: opts_GNFA
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
    subopts: opts_GNFA
    ...
```
"""
function DDVFA(opts::opts_DDVFA)
    subopts = opts_GNFA(
        rho=opts.rho_ub,
        display=false
    )
    DDVFA(opts,
          subopts,
          DataConfig(),
          0.0,
          Array{GNFA}(undef, 0),
          Array{Int}(undef, 0),
          Array{Float}(undef, 0, 0),
          Array{Float}(undef, 0, 0),
          0,
          0,
          0.0,
          0.0
    )
end # DDVFA(opts::opts_DDVFA)

"""
    train!(art::DDVFA, x::RealArray ; y::IntegerVector=Vector{Int}(), preprocessed::Bool=false)

Train the DDVFA model on the data.
"""
function train!(art::DDVFA, x::RealArray ; y::IntegerVector = Vector{Int}(), preprocessed::Bool=false)
    # Show a message if display is on
    art.opts.display && @info "Training DDVFA"

    # Simple supervised flag
    supervised = !isempty(y)

    # Data information and setup
    n_samples = get_n_samples(x)

    # Set up the data config if training for the first time
    !art.config.setup && data_setup!(art.config, x)

    # If the data is not preprocessed, then complement code it
    if !preprocessed
        x = complement_code(x, config=art.config)
    end

    # art.labels = zeros(n_samples)
    if n_samples == 1
        y_hat = zero(Int)
    else
        y_hat = zeros(Int, n_samples)
    end

    # Initialization
    if isempty(art.F2)
        # Set the first label as either 1 or the first provided label
        local_label = supervised ? y[1] : 1
        # Add the local label to the output vector
        if n_samples == 1
            y_hat = local_label
        else
            y_hat[1] = local_label
        end
        # Create a new category
        create_category(art, get_sample(x, 1), local_label)
        # Skip the first training entry
        skip_first = true
    else
        skip_first = false
    end

    # Initialize old weight vector for checking stopping conditions between epochs
    art.W_old = deepcopy(art.W)

    # Set the learning threshold as a function of the data dimension
    art.threshold = art.opts.rho*(art.config.dim^art.opts.gamma_ref)

    # Learn until the stopping conditions
    art.epoch = 0
    while true
        # Increment the epoch and get the iterator
        art.epoch += 1
        iter = get_iterator(art.opts, x)
        for i = iter
            # Update the iterator if necessary
            update_iter(art, iter, i)
            # Skip the first sample if we just initialized
            (i == 1 && skip_first) && continue
            # Grab the sample slice
            sample = get_sample(x, i)

            # Default to mismatch
            mismatch_flag = true
            # If label is new, break to make new category
            if supervised && !(y[i] in art.labels)
                if n_samples == 1
                    y_hat = y[i]
                else
                    y_hat[i] = y[i]
                end
                create_category(art, sample, y[i])
                continue
            end
            # Otherwise, check for match
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
                    if supervised && art.labels[bmu] != y[i]
                        break
                    end
                    # Update the weights with the sample
                    train!(art.F2[bmu], sample)
                    # Save the output label for the sample
                    label = art.labels[bmu]
                    if n_samples == 1
                        y_hat = label
                    else
                        y_hat[i] = label
                    end
                    mismatch_flag = false
                    break
                end
            end
            if mismatch_flag
                # Update the stored match and activation values
                bmu = index[1]
                art.M = similarity(art.opts.method, art.F2[bmu], "M", sample, art.opts.gamma_ref)
                art.T = T[bmu]
                # Get the correct label
                label = supervised ? y[i] : art.n_categories + 1
                if n_samples == 1
                    y_hat = label
                else
                    y_hat[i]  = label
                end
                create_category(art, sample, label)
            end
        end
        # Make sure to start at first sample from now on
        skip_first = false
        # Deep copy all of the weights for stopping condition check
        art.W = art.F2[1].W
        for kx = 2:art.n_categories
            art.W = [art.W art.F2[kx].W]
        end
        if stopping_conditions(art)
            break
        end
        art.W_old = deepcopy(art.W)
    end
    return y_hat
end # train!(art::DDVFA, x::RealArray ; y::IntegerVector = Vector{Int}(), preprocessed::Bool=false)

"""
    create_category(art::DDVFA, sample::RealVector, label::Integer)

Create a new category by appending and initializing a new GNFA node to F2.
"""
function create_category(art::DDVFA, sample::RealVector, label::Integer)
    # Global Fuzzy ART
    art.n_categories += 1
    push!(art.labels, label)
    # Local Fuzzy ART
    push!(art.F2, GNFA(art.subopts, sample))
end # function create_category(art::DDVFA, sample::RealVector, label::Integer)

"""
    stopping_conditions(art::DDVFA)

Stopping conditions for Distributed Dual Vigilance Fuzzy ARTMAP.

Returns true if there is no change in weights during the epoch or the maxmimum epochs has been reached.
"""
function stopping_conditions(art::DDVFA)
    # Compute the stopping condition, return a bool
    return art.W == art.W_old || art.epoch >= art.opts.max_epoch
end # stopping_conditions(DDVFA)

"""
    similarity(method::String, F2::GNFA, field_name::String, sample::RealVector, gamma_ref::RealFP)

Compute the similarity metric depending on method with explicit comparisons
for the field name.
"""
function similarity(method::String, F2::GNFA, field_name::String, sample::RealVector, gamma_ref::RealFP)
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
end # similarity(method::String, F2::GNFA, field_name::String, sample::RealVector, gamma_ref::RealFP)

"""
    classify(art::DDVFA, x::RealArray ; preprocessed::Bool=false, get_bmu::Bool=false)

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
function classify(art::DDVFA, x::RealArray ; preprocessed::Bool=false, get_bmu::Bool=false)
    # Show a message if display is on
    art.opts.display && @info "Testing DDVFA"

    # Data information and setup
    n_samples = get_n_samples(x)

    # Verify that the data is setup before classifying
    !art.config.setup && @error "Attempting to classify data before setup"

    # If the data is not preprocessed, then complement code it
    if !preprocessed
        x = complement_code(x, config=art.config)
    end

    # Initialize the output vector
    if n_samples == 1
        y_hat = zero(Int)
    else
        y_hat = zeros(Int, n_samples)
    end

    # Get the iterator based on the module options and data shape
    iter = get_iterator(art.opts, x)
    for ix = iter
        # Update the iterator if necessary
        update_iter(art, iter, ix)

        # Grab the sample slice
        sample = get_sample(x, ix)

        # Calculate all global activations
        T = zeros(art.n_categories)
        for jx = 1:art.n_categories
            activation_match!(art.F2[jx], sample)
            T[jx] = similarity(art.opts.method, art.F2[jx], "T", sample, art.opts.gamma_ref)
        end
        # Sort by highest activation
        index = sortperm(T, rev=true)

        mismatch_flag = true
        for jx = 1:art.n_categories
            bmu = index[jx]
            M = similarity(art.opts.method, art.F2[bmu], "M", sample, art.opts.gamma_ref)
            if M >= art.threshold
                # Update the stored match and activation values
                art.M = M
                art.T = T[bmu]
                # Current winner
                label = art.labels[bmu]
                if n_samples == 1
                    y_hat = label
                else
                    y_hat[ix] = label
                end
                mismatch_flag = false
                break
            end
        end
        if mismatch_flag
            @debug "Mismatch"
            # Update the stored match and activation values
            bmu = index[1]
            art.M = similarity(art.opts.method, art.F2[bmu], "M", sample, art.opts.gamma_ref)
            art.T = T[bmu]
            # If falling back to the highest activated category, return that
            if get_bmu
                label = art.labels[index[1]]
                if n_samples == 1
                    y_hat = label
                else
                    y_hat[ix] = label
                end
            # Otherwise, return a mismatch
            else
                if n_samples == 1
                    y_hat = -1
                else
                    y_hat[ix] = -1
                end
            end
        end
    end

    return y_hat
end # classify(art::DDVFA, x::RealArray ; preprocessed::Bool=false, get_bmu::Bool=false)
