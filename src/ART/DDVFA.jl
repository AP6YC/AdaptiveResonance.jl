"""
    DDVFA.jl

Description:
    Includes all of the structures and logic for running a Distributed Dual-Vigilance Fuzzy ART (DDVFA) module.
"""

"""
    opts_GNFA()

Gamma-Normalized Fuzzy ART options struct.

# Examples
```julia-repl
julia> opts_GNFA()
Initialized GNFA
```
"""
@with_kw mutable struct opts_GNFA <: ARTOpts @deftype RealFP
    # Vigilance parameter: [0, 1]
    rho = 0.6; @assert rho >= 0.0 && rho <= 1.0
    # Choice parameter: alpha > 0
    alpha = 1e-3; @assert alpha > 0.0
    # Learning parameter: (0, 1]
    beta = 1.0; @assert beta > 0.0 && beta <= 1.0
    # "Pseudo" kernel width: gamma >= 1
    gamma = 3.0; @assert gamma >= 1.0
    # gamma = 784; @assert gamma >= 1
    # "Reference" gamma for normalization: 0 <= gamma_ref < gamma
    gamma_ref = 1.0; @assert 0.0 <= gamma_ref && gamma_ref < gamma
    # Similarity method (activation and match):
    #   'single', 'average', 'complete', 'median', 'weighted', or 'centroid'
    method::String = "single"
    # Display flag
    display::Bool = true
    # Maximum number of epochs during training
    max_epochs::Integer = 1
end # opts_GNFA

"""
    GNFA <: ART

Gamma-Normalized Fuzzy ART learner struct

# Examples
```julia-repl
julia> GNFA()
GNFA
    opts: opts_GNFA
    ...
```
"""
mutable struct GNFA <: ART
    # Assign numerical parameters from options
    opts::opts_GNFA
    config::DataConfig

    # Working variables
    threshold::RealFP
    labels::IntegerVector
    T::RealVector
    M::RealVector

    # "Private" working variables
    W::RealMatrix
    W_old::RealMatrix
    n_instance::IntegerVector
    n_categories::Integer
    epoch::Integer
end # GNFA <: ART

"""
    GNFA()

Implements a Gamma-Normalized Fuzzy ART learner.

# Examples
```julia-repl
julia> GNFA()
GNFA
    opts: opts_GNFA
    ...
```
"""
function GNFA()
    opts = opts_GNFA()
    GNFA(opts)
end # GNFA()

"""
    GNFA(opts::opts_GNFA)

Implements a Gamma-Normalized Fuzzy ART learner with specified options.

# Examples
```julia-repl
julia> GNFA(opts)
GNFA
    opts: opts_GNFA
    ...
```
"""
function GNFA(opts::opts_GNFA)
    GNFA(opts,                          # opts
         DataConfig(),                  # config
         0.0,                           # threshold
         Array{Integer}(undef,0),       # labels
         Array{RealFP}(undef, 0),       # T
         Array{RealFP}(undef, 0),       # M
         Array{RealFP}(undef, 0, 0),    # W
         Array{RealFP}(undef, 0, 0),    # W_old
         Array{Integer}(undef, 0),      # n_instance
         0,                             # n_categories
         0                              # epoch
    )
end # GNFA(opts::opts_GNFA)

"""
    GNFA(opts::opts_GNFA, sample::RealArray)

Create and initialize a GNFA with a single sample in one step.
"""
function GNFA(opts::opts_GNFA, sample::RealArray)
    art = GNFA(opts)
    initialize!(art, sample)
    return art
end # GNFA(opts::opts_GNFA, sample::RealArray)

"""
    initialize!(art::GNFA, x::Array)

Initializes a GNFA learner with an intial sample 'x'.

# Examples
```julia-repl
julia> my_GNFA = GNFA()
GNFA
    opts: opts_GNFA
    ...
julia> initialize!(my_GNFA, [1 2 3 4])
```
"""
# function initialize!(art::GNFA, x::RealArray ; y::Integer=0)
function initialize!(art::GNFA, x::Vector{T} ; y::Integer=0) where {T<:RealFP}
    # Set up the data config
    if art.config.setup
        @warn "Data configuration already set up, overwriting config"
    else
        art.config.setup = true
    end

    # IMPORTANT: Assuming that x is a sample, so each entry is a feature
    dim = length(x)
    art.config.dim_comp = dim
    art.config.dim = Integer(dim/2) # Assumes input is already complement coded

    # Initialize the instance and categories counters
    art.n_instance = [1]
    art.n_categories = 1

    # Set the threshold
    art.threshold = art.opts.rho * (art.config.dim^art.opts.gamma_ref)
    # Fast commit the weight
    art.W = Array{T}(undef, art.config.dim_comp, 1)
    # Assign the contents, valid this way for 1-D or 2-D arrays
    art.W[:, 1] = x
    label = y == 0 ? y : 1
    push!(art.labels, label)
end # initialize!(art::GNFA, x::RealArray ; y::Integer=0)

"""
    train!(art::GNFA, x::RealArray ; y::IntegerVector=[])

Trains a GNFA learner with dataset 'x' and optional labels 'y'

# Examples
```julia-repl
julia> my_GNFA = GNFA()
GNFA
    opts: opts_GNFA
    ...
julia> x = load_data()
julia> train!(my_GNFA, x)
```
"""
function train!(art::GNFA, x::RealArray ; y::IntegerVector = Vector{Integer}())
    # Flag for if training in supervised mode
    supervised = !isempty(y)
    # Initialization if weights are empty; fast commit the first sample
    if isempty(art.W)
        label = supervised ? y[1] : 1
        push!(art.labels, label)
        initialize!(art, x[:, 1])
        skip_first = true
    else
        skip_first = false
    end

    art.W_old = deepcopy(art.W)

    # Learning
    art.epoch = 0
    while true
        # Increment the epoch and get the iterator
        art.epoch += 1
        iter = get_iterator(art.opts, x)
        # Loop over samples
        for i = iter
            # Update the iterator if necessary
            update_iter(art, iter, i)
            # Skip the first sample if we just initialized
            (i == 1 && skip_first) && continue
            # Grab the sample slice
            sample = get_sample(x, i)
            # Compute activation/match functions
            activation_match!(art, sample)
            # Sort activation function values in descending order
            index = sortperm(art.T, rev=true)
            # Initialize mismatch as true
            mismatch_flag = true
            # Loop over all categories
            for j = 1:art.n_categories
                # Best matching unit
                bmu = index[j]
                # Vigilance check - pass
                if art.M[bmu] >= art.threshold
                    # Learn the sample
                    learn!(art, sample, bmu)
                    # Update sample labels
                    label = supervised ? y[i] : bmu
                    push!(art.labels, label)
                    # No mismatch
                    mismatch_flag = false
                    break
                end
            end
            # If there was no resonant category, make a new one
            if mismatch_flag
                # Increment the number of categories
                art.n_categories += 1
                # Fast commit
                # art.W = [art.W x[:, i]]
                art.W = hcat(art.W, sample)
                # Increment number of samples associated with new category
                push!(art.n_instance, 1)
                # Update sample labels
                label = supervised ? y[i] : art.n_categories
                push!(art.labels, label)
            end
        end
        # Make sure to start at first sample from now on
        skip_first = false
        # Check for the stopping condition for the whole loop
        if stopping_conditions(art)
            break
        end
        # If we didn't break, deep copy the old weights
        art.W_old = deepcopy(art.W)
    end
end # train!(art::GNFA, x::RealArray ; y::IntegerVector = Vector{Integer}())

"""
    classify(art::GNFA, x::RealArray)

Predict categories of 'x' using the GNFA model.

Returns predicted categories 'y_hat'

# Examples
```julia-repl
julia> my_GNFA = GNFA()
GNFA
    opts: opts_GNFA
    ...
julia> x, y = load_data()
julia> train!(my_GNFA, x)
julia> y_hat = classify(my_GNFA, y)
```
"""
function classify(art::GNFA, x::RealArray)
    # Get the number of samples to classify
    n_samples = get_n_samples(x)

    # Initialize the output vector and iterate across all data
    y_hat = zeros(Integer, n_samples)
    iter = get_iterator(art.opts, x)
    for ix in iter
        # Update the iterator if necessary
        update_iter(art, iter, ix)
        # Compute activation and match functions
        activation_match!(art, x[:, ix])
        # Sort activation function values in descending order
        index = sortperm(art.T, rev=true)
        mismatch_flag = true
        for jx in 1:art.n_categories
            bmu = index[jx]
            # Vigilance check - pass
            if art.M[bmu] >= art.threshold
                # Current winner
                y_hat[ix] = art.labels[bmu]
                mismatch_flag = false
                break
            end
        end
        if mismatch_flag
            # Create new weight vector
            @debug "Mismatch"
            y_hat[ix] = -1
        end
    end
    return y_hat
end # classify(art::GNFA, x::RealArray)

"""
    activation_match!(art::GNFA, x::RealArray)

Computes the activation and match functions of the art module against sample x.

# Examples
```julia-repl
julia> my_GNFA = GNFA()
GNFA
    opts: opts_GNFA
    ...
julia> x, y = load_data()
julia> train!(my_GNFA, x)
julia> x_sample = x[:, 1]
julia> activation_match!(my_GNFA, x_sample)
```
"""
function activation_match!(art::GNFA, x::RealArray)
    art.T = zeros(art.n_categories)
    art.M = zeros(art.n_categories)
    for i = 1:art.n_categories
        W_norm = norm(art.W[:, i], 1)
        art.T[i] = (norm(element_min(x, art.W[:, i]), 1)/(art.opts.alpha + W_norm))^art.opts.gamma
        art.M[i] = (W_norm^art.opts.gamma_ref)*art.T[i]
    end
end # activation_match!(art::GNFA, x::RealArray)

"""
    learn(art::GNFA, x::RealVector, W::RealVector)

Return the modified weight of the art module conditioned by sample x.
"""
function learn(art::GNFA, x::RealVector, W::RealVector)
    # Update W
    return art.opts.beta .* element_min(x, W) .+ W .* (1 - art.opts.beta)
end # learn(art::GNFA, x::RealVector, W::RealVector)

"""
    learn!(art::GNFA, x::RealVector, index::Integer)

In place learning function with instance counting.
"""
function learn!(art::GNFA, x::RealVector, index::Integer)
    # Update W
    art.W[:, index] = learn(art, x, art.W[:, index])
    art.n_instance[index] += 1
end # learn!(art::GNFA, x::RealVector, index::Integer)

"""
    stopping_conditions(art::GNFA)

Stopping conditions for a GNFA module.
"""
function stopping_conditions(art::GNFA)
    return isequal(art.W, art.W_old) || art.epoch >= art.opts.max_epochs
end # stopping_conditions(art::GNFA)

"""
    opts_DDVFA()

Distributed Dual Vigilance Fuzzy ART options struct.

# Examples
```julia-repl
julia> my_opts = opts_DDVFA()
```
"""
@with_kw mutable struct opts_DDVFA <: ARTOpts @deftype RealFP
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
    max_epoch::Integer = 1
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
    threshold::RealFP
    F2::Vector{GNFA}
    labels::IntegerVector
    W::RealMatrix        # All F2 nodes' weight vectors
    W_old::RealMatrix    # Old F2 node weight vectors (for stopping criterion)
    n_categories::Integer
    epoch::Integer
    T::RealFP
    M::RealFP
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
          Array{Integer}(undef, 0),
          Array{RealFP}(undef, 0, 0),
          Array{RealFP}(undef, 0, 0),
          0,
          0,
          0.0,
          0.0
    )
end # DDVFA(opts::opts_DDVFA)

"""
    train!(art::DDVFA, x::RealArray ; y::IntegerVector=[], preprocessed::Bool=false)

Train the DDVFA model on the data.
"""
function train!(art::DDVFA, x::RealArray ; y::IntegerVector = Vector{Integer}(), preprocessed::Bool=false)
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
        y_hat = zero(Integer)
    else
        y_hat = zeros(Integer, n_samples)
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
end # train!(art::DDVFA, x::RealArray ; y::IntegerVector = Vector{Integer}(), preprocessed::Bool=false)

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
        y_hat = zero(Integer)
    else
        y_hat = zeros(Integer, n_samples)
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
