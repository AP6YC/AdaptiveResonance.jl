"""
    DVFA.jl

Description:
    Includes all of the structures and logic for running a Dual-Vigilance Fuzzy ART (DVFA) module.

Authors:
    MATLAB implementation: Leonardo Enzo Brito da Silva
    Julia port: Sasha Petrenko <sap625@mst.edu>

References:
[1] L. E. Brito da Silva, I. Elnabarawy and D. C. Wunsch II, "Dual
Vigilance Fuzzy ART," Neural Networks Letters. To appear.
[2] G. Carpenter, S. Grossberg, and D. Rosen, "Fuzzy ART: Fast
stable learning and categorization of analog patterns by an adaptive
resonance system," Neural Networks, vol. 4, no. 6, pp. 759–771, 1991.
"""

"""
    opts_DVFA()

Dual Vigilance Fuzzy ART options struct.

# Examples
```julia-repl
julia> my_opts = opts_DVFA()
```
"""
@with_kw mutable struct opts_DVFA <: ARTOpts @deftype RealFP
    # Lower-bound vigilance parameter: [0, 1]
    rho_lb = 0.55; @assert rho_lb >= 0.0 && rho_lb <= 1.0
    # Upper bound vigilance parameter: [0, 1]
    rho_ub = 0.75; @assert rho_ub >= 0.0 && rho_ub <= 1.0
    # Choice parameter: alpha > 0
    alpha = 1e-3; @assert alpha > 0.0
    # Learning parameter: (0, 1]
    beta = 1.0; @assert beta > 0.0 && beta <= 1.0
    # Display flag
    display::Bool = true
    # Maximum number of epochs during training
    max_epochs::Integer = 1
end # opts_DVFA

"""
    DVFA <: ART

Dual Vigilance Fuzzy ARTMAP module struct.

# Examples
```julia-repl
julia> DVFA()
DVFA
    opts: opts_DVFA
    ...
```
"""
mutable struct DVFA <: ART
    # Get parameters
    opts::opts_DVFA
    config::DataConfig

    # Working variables
    labels::IntegerVector
    W::RealMatrix
    T::RealVector
    M::RealVector
    W_old::RealMatrix
    map::IntegerVector
    n_categories::Integer
    n_clusters::Integer
    epoch::Integer
end # DVFA

"""
    DVFA()

Implements a DVFA learner with default options.

# Examples
```julia-repl
julia> DVFA()
DVFA
    opts: opts_DDVFA
    ...
```
"""
function DVFA()
    opts = opts_DVFA()
    DVFA(opts)
end # DVFA()

"""
    DVFA(;kwargs...)

Implements a DVFA learner with keyword arguments.

# Examples
```julia-repl
julia> DVFA(rho=0.7)
DVFA
    opts: opts_DDVFA
    ...
```
"""
function DVFA(;kwargs...)
    opts = opts_DVFA(;kwargs...)
    DVFA(opts)
end # DVFA(;kwargs...)

"""
    DVFA(opts::opts_DVFA)

Implements a DVFA learner with specified options.

# Examples
```julia-repl
julia> my_opts = opts_DVFA()
julia> DVFA(my_opts)
DVFA
    opts: opts_DVFA
    ...
```
"""
function DVFA(opts::opts_DVFA)
    DVFA(
        opts,                           # opts
        DataConfig(),                   # config
        Array{Integer}(undef, 0),       # labels
        Array{RealFP}(undef, 0, 0),     # W
        Array{RealFP}(undef, 0),        # M
        Array{RealFP}(undef, 0),        # T
        Array{RealFP}(undef, 0, 0),     # W_old
        Array{Integer}(undef, 0),       # map
        0,                              # n_categories
        0,                              # n_clusters
        0                               # epoch
    )
end # DDVFA(opts::opts_DDVFA)

"""
    train!(art::DVFA, x::RealArray ; y::IntegerVector = [], preprocessed::Bool=false)

Train the DVFA module on x with optional custom category labels y.

# Arguments
- `art::DVFA`: the dual-vigilance fuzzy art module to train.
- `x::RealArray`: the data to train on, interpreted as a single sample if x is a vector.
- `y::IntegerVector=[]`: optional custom labels to assign to the categories. If empty, ordinary incremental labels are prescribed.
"""
function train!(art::DVFA, x::RealArray ; y::IntegerVector = Vector{Integer}(), preprocessed::Bool=false)
    # Show a message if display is on
    art.opts.display && @info "Training DVFA"

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

    if n_samples == 1
        y_hat = zero(Integer)
    else
        y_hat = zeros(Integer, n_samples)
    end

    # Initialization
    if isempty(art.W)
        # Set the first label as either 1 or the first provided label
        local_label = supervised ? y[1] : 1
        # Add the local label to the output vector
        if n_samples == 1
            y_hat = local_label
        else
            y_hat[1] = local_label
        end
        # Create a new category and cluster
        art.W = ones(art.config.dim_comp, 1)
        art.n_categories = 1
        art.n_clusters = 1
        push!(art.labels, local_label)
        # Skip the first training entry
        skip_first = true
    else
        skip_first = false
    end
    art.W_old = art.W

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

            # If label is new, break to make new category
            if supervised && !(y[i] in art.labels)
                if n_samples == 1
                    y_hat = y[i]
                else
                    y_hat[i] = y[i]
                end
                # Update sample labels
                push!(art.labels, y[i])
                # Fast commit the sample
                art.W = hcat(art.W, sample)
                art.n_categories += 1
                art.n_clusters += 1
                continue
            end

            # Compute the activation and match for all categories
            activation_match!(art, sample)
            # Sort activation function values in descending order
            index = sortperm(art.T, rev=true)
            # Default to mismatch
            mismatch_flag = true
            # Loop over all categories
            for j = 1:art.n_categories
                # Best matching unit
                bmu = index[j]
                # Vigilance test upper bound
                if art.M[bmu] >= art.opts.rho_ub * art.config.dim
                    # Learn the sample
                    learn!(art, sample, bmu)
                    # Update sample label for output`
                    label = supervised ? y[i] : art.labels[bmu]
                    # push!(art.labels, label)
                    # No mismatch
                    mismatch_flag = false
                    break
                # Vigilance test lower bound
                elseif art.M[bmu] >= art.opts.rho_lb * art.config.dim
                    # # Update sample labels
                    # label = supervised ? y[i] : art.map[bmu]
                    label = supervised ? y[i] : art.labels[bmu]
                    push!(art.labels, label)
                    # Fast commit the sample
                    art.W = hcat(art.W, sample)
                    art.n_categories += 1
                    # No mismatch
                    mismatch_flag = false
                    break
                end
            end
            # If there was no resonant category, make a new one
            if mismatch_flag
                # Create a new category-to-cluster label
                # push!(art.map, last(art.map) + 1)
                label = supervised ? y[i] : art.n_clusters + 1
                push!(art.labels, label)
                # Fast commit the sample
                art.W = hcat(art.W, sample)
                # Increment the number of categories and clusters
                art.n_categories += 1
                art.n_clusters += 1
            end

            if n_samples == 1
                y_hat = label
            else
                y_hat[i] = label
            end
        end
        # Check for the stopping condition for the whole loop
        if stopping_conditions(art)
            break
        end
    end

    return y_hat
end # train!(art::DVFA, x::RealArray ; y::IntegerVector = Vector{Integer}(), preprocessed::Bool=false)

"""
    classify(art::DVFA, x::RealArray)

Predict categories of 'x' using the DVFA model.

Returns predicted categories 'y_hat'

# Examples
```julia-repl
julia> my_DVFA = DVFA()
DVFA
    opts: opts_DVFA
    ...
julia> x, y = load_data()
julia> train!(my_DVFA, x)
julia> y_hat = classify(my_DVFA, y)
```
"""
function classify(art::DVFA, x::RealArray ; preprocessed::Bool=false, get_bmu::Bool=false)
    # Show a message if display is on
    art.opts.display && @info "Testing DVFA"

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
            if art.M[bmu] >= art.opts.rho_ub * art.config.dim
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
            # Create new weight vector
            @debug "Mismatch"
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
end # classify(art::DVFA, x::RealArray ; preprocessed::Bool=false, get_bmu::Bool=false)

"""
    activation_match!(art::DVFA, x::RealVector)

Compute and store the activation and match values for the DVFA module.
"""
function activation_match!(art::DVFA, x::RealVector)
    art.T = zeros(art.n_categories)
    art.M = zeros(art.n_categories)
    for jx = 1:art.n_categories
        numerator = norm(element_min(x, art.W[:, jx]), 1)
        art.T[jx] = numerator/(art.opts.alpha + norm(art.W[:, jx], 1))
        art.M[jx] = numerator
    end # activation_match!(art::DVFA, x::RealVector)
end # activation_match!(art::DVFA, x::RealVector)

"""
    learn(art::DVFA, x::RealVector, W::RealVector)

Return the modified weight of the DVFA module conditioned by sample x.
"""
function learn(art::DVFA, x::RealVector, W::RealVector)
    # Update W
    return art.opts.beta .* element_min(x, W) .+ W .* (1 - art.opts.beta)
end # learn(art::DVFA, x::RealVector, W::RealVector)

"""
    learn!(art::DVFA, x::RealVector, index::Integer)

In place learning function.
"""
function learn!(art::DVFA, x::RealVector, index::Integer)
    # Update W
    art.W[:, index] = learn(art, x, art.W[:, index])
end # learn!(art::DVFA, x::RealVector, index::Integer)

"""
    stopping_conditions(art::DVFA)

Stopping conditions for a DVFA module.
"""
function stopping_conditions(art::DVFA)
    return isequal(art.W, art.W_old) || art.epoch >= art.opts.max_epochs
end # stopping_conditions(art::DVFA)
