"""
    DAM.jl

Description:
    Options, structures, and logic for the Default ARTMAP (DAM) module.
"""

"""
    opts_DAM()

Implements a Default ARTMAP learner's options.

# Examples
```julia-repl
julia> my_opts = opts_DAM()
```
"""
@with_kw mutable struct opts_DAM <: ARTOpts @deftype RealFP
    # Vigilance parameter: [0, 1]
    rho = 0.6; @assert rho >= 0.0 && rho <= 1.0
    # Choice parameter: alpha > 0
    alpha = 1e-7; @assert alpha > 0.0
    # Match tracking parameter
    epsilon = -1e-3; @assert epsilon > -1.0 && epsilon < 1.0
    # Learning parameter: (0, 1]
    beta = 1.0; @assert beta > 0.0 && beta <= 1.0
    # Display flag
    display::Bool = true
    # Maximum number of epochs during training
    max_epochs::Integer = 1
end # opts_DAM()

"""
    DAM <: ARTMAP

Default ARTMAP struct.
"""
mutable struct DAM <: ARTMAP
    opts::opts_DAM
    config::DataConfig
    W::RealMatrix
    W_old::RealMatrix
    labels::IntegerVector
    y::IntegerVector
    n_categories::Integer
    epoch::Integer
end # DAM <: ARTMAP

"""
    DAM()

Implements a Default ARTMAP learner.

# Examples
```julia-repl
julia> DAM()
DAM
    opts: opts_DAM
    ...
```
"""
function DAM()
    opts = opts_DAM()
    DAM(opts)
end # DAM()

"""
    DAM(opts)

Implements a Default ARTMAP learner with specified options

# Examples
```julia-repl
julia> opts = opts_DAM()
julia> DAM(opts)
DAM
    opts: opts_DAM
    ...
```
"""
function DAM(opts::opts_DAM)
    DAM(opts,                       # opts_DAM
        DataConfig(),               # config
        Array{RealFP}(undef, 0,0), # W
        Array{RealFP}(undef, 0,0), # W_old
        Array{Integer}(undef, 0),       # labels
        Array{Integer}(undef, 0),       # y
        0,                          # n_categories
        0                           # epoch
    )
end # DAM(opts::opts_DAM)

"""
    train!(art::DAM, x::RealArray, y::RealArray ; preprocessed::Bool=false)

Trains a Default ARTMAP learner in a supervised manner.

# Examples
```julia-repl
julia> x, y = load_data()
julia> art = DAM()
DAM
    opts: opts_DAM
    ...
julia> train!(art, x, y)
```
"""
function train!(art::DAM, x::RealArray, y::RealArray ; preprocessed::Bool=false)
    # Show a message if display is on
    art.opts.display && @info "Training DAM"

    # Data information and setup
    n_samples = get_n_samples(x)

    # Set up the data config if it is not already
    !art.config.setup && data_setup!(art.config, x)

    # If the data isn't preprocessed, then complement code it with the config
    if !preprocessed
        x = complement_code(x, config=art.config)
    end

    # Convenient semantic flag
    # is_supervised = !isempty(y)

    # Initialize the internal categories
    art.y = zeros(Integer, n_samples)

    # Initialize the training loop, continue to convergence
    art.epoch = 0
    while true
        # Increment the epoch and get the iterator
        art.epoch += 1
        iter = get_iterator(art.opts, x)
        for ix in iter
            # Update the iterator if necessary
            update_iter(art, iter, ix)
            if !(y[ix] in art.labels)
                # Initialize W and labels
                if isempty(art.W)
                    art.W = Array{Float64}(undef, art.config.dim_comp, 1)
                    art.W_old = Array{Float64}(undef, art.config.dim_comp, 1)
                    art.W[:, ix] = x[:, ix]
                else
                    art.W = [art.W x[:, ix]]
                end
                push!(art.labels, y[ix])
                art.n_categories += 1
                art.y[ix] = y[ix]
            else
                # Baseline vigilance parameter
                rho_baseline = art.opts.rho

                # Compute activation function
                T = zeros(art.n_categories)
                for jx in 1:art.n_categories
                    T[jx] = activation(art, x[:, ix], art.W[:, jx])
                end

                # Sort activation function values in descending order
                index = sortperm(T, rev=true)
                mismatch_flag = true
                for jx in 1:art.n_categories
                    # Compute match function
                    M = art_match(art, x[:, ix], art.W[:, index[jx]])
                    @debug M
                    # Current winner
                    if M >= rho_baseline
                        if y[ix] == art.labels[index[jx]]
                            # Learn
                            @debug "Learning"
                            art.W[:, index[jx]] = learn(art, x[:, ix], art.W[:, index[jx]])
                            art.y[ix] = art.labels[index[jx]]
                            mismatch_flag = false
                            break
                        else
                            # Match tracking
                            @debug "Match tracking"
                            rho_baseline = M + art.opts.epsilon
                        end
                    end
                end
                if mismatch_flag
                    # Create new weight vector
                    @debug "Mismatch"
                    art.W = hcat(art.W, x[:, ix])
                    push!(art.labels, y[ix])
                    art.n_categories += 1
                    art.y[ix] = y[ix]
                end
            end
        end
        if stopping_conditions(art)
            break
        end
        art.W_old = deepcopy(art.W)
    end
end # train!(art::DAM, x::RealArray, y::RealArray ; preprocessed::Bool=false)

"""
    classify(art::DAM, x::RealArray ; preprocessed::Bool=false)

Categorize data 'x' using a trained Default ARTMAP module 'art'.

# Examples
```julia-repl
julia> x, y = load_data()
julia> x_test, y_test = load_test_data()
julia> art = DAM()
DAM
    opts: opts_DAM
    ...
julia> train!(art, x, y)
julia> classify(art, x_test)
```
"""
function classify(art::DAM, x::RealArray ; preprocessed::Bool=false)
    # Show a message if display is on
    art.opts.display && @info "Testing DAM"

    # Data information and setup
    n_samples = get_n_samples(x)

    # Throw an soft error if classifying before setup
    !art.config.setup && @error "Attempting to classify data before setup"

    # If the data is not preprocessed, then complement code it
    if !preprocessed
        x = complement_code(x, config=art.config)
    end

    # Initialize the output vector and iterate across all data
    y_hat = zeros(Int, n_samples)
    iter = get_iterator(art.opts, x)
    for ix in iter
        # Update the iterator if necessary
        update_iter(art, iter, ix)

        # Compute activation function
        T = zeros(art.n_categories)
        for jx in 1:art.n_categories
            T[jx] = activation(art, x[:, ix], art.W[:, jx])
        end

        # Sort activation function values in descending order
        index = sortperm(T, rev=true)
        mismatch_flag = true
        for jx in 1:art.n_categories
            # Compute match function
            M = art_match(art, x[:, ix], art.W[:, index[jx]])
            @debug M
            # Current winner
            if M >= art.opts.rho
                y_hat[ix] = art.labels[index[jx]]
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
end # classify(art::DAM, x::RealArray ; preprocessed::Bool=false)

"""
    stopping_conditions(art::DAM)

Stopping conditions for Default ARTMAP, checked at the end of every epoch.
"""
function stopping_conditions(art::DAM)
    # Compute the stopping condition, return a bool
    return art.W == art.W_old || art.epoch >= art.opts.max_epochs
end # stopping_conditions(art::DAM)

"""
    activation(art::DAM, x::RealVector, W::RealVector)

Default ARTMAP's choice-by-difference activation function.
"""
function activation(art::DAM, x::RealVector, W::RealVector)
    # Compute T and return
    return norm(element_min(x, W), 1) +
        (1-art.opts.alpha)*(art.config.dim - norm(W, 1))
end # activation(art::DAM, x::RealVector, W::RealVector)

"""
    learn(art::DAM, x::RealVector, W::RealVector)

Returns a single updated weight for the Simple Fuzzy ARTMAP module for weight
vector W and sample x.
"""
function learn(art::DAM, x::RealVector, W::RealVector)
    # Update W
    return art.opts.beta .* element_min(x, W) .+ W .* (1 - art.opts.beta)
end # learn(art::DAM, x::RealVector, W::RealVector)

"""
    art_match(art::DAM, x::RealVector, W::RealVector)

Returns the match function for the Default ARTMAP module with weight W and
sample x.
"""
function art_match(art::DAM, x::RealVector, W::RealVector)
    # Compute M and return
    return norm(element_min(x, W), 1) / art.config.dim
end # art_match(art::DAM, x::RealVector, W::RealVector)
