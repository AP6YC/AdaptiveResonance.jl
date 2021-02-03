"""
    opts_SFAM()

Implements a Simple Fuzzy ARTMAP learner's options.

# Examples
```julia-repl
julia> my_opts = opts_SFAM()
```
"""
@with_kw mutable struct opts_SFAM <: AbstractARTOpts @deftype Float64
    # Vigilance parameter: [0, 1]
    rho = 0.75; @assert rho >= 0 && rho <= 1
    # Choice parameter: alpha > 0
    alpha = 1e-7; @assert alpha > 0
    # Match tracking parameter
    epsilon = 1e-3; @assert epsilon > 0 && epsilon < 1
    # Learning parameter: (0, 1]
    beta = 1; @assert beta > 0 && beta <= 1
    # Uncommitted node flag
    uncommitted::Bool = true
    # Display flag
    display::Bool = true
    # shuffle::Bool = false
    random_seed = 1234.5678
    max_epochs = 1
end # opts_SFAM

"""
    SFAM

Simple Fuzzy ARTMAP struct.
"""
mutable struct SFAM <: AbstractART
    opts::opts_SFAM
    W::Array{Float64, 2}
    W_old::Array{Float64, 2}
    labels::Array{Int, 1}
    y::Array{Int, 1}
    dim::Int
    n_categories::Int
    epoch::Int
end

"""
    SFAM()

Implements a Simple Fuzzy ARTMAP learner.

# Examples
```julia-repl
julia> SFAM()
SFAM
    opts: opts_SFAM
    ...
```
"""
function SFAM()
    opts = opts_SFAM()
    SFAM(opts)
end

"""
    SFAM(opts)

Implements a Simple Fuzzy ARTMAP learner with specified options.

# Examples
```julia-repl
julia> opts = opts_SFAM()
julia> SFAM(opts)
SFAM
    opts: opts_SFAM
    ...
```
"""
function SFAM(opts::opts_SFAM)
    SFAM(opts,                          # opts_SFAM
         Array{Float64}(undef, 0,0),    # W
         Array{Float64}(undef, 0,0),    # W_old
         Array{Int}(undef, 0),          # labels
         Array{Int}(undef, 0),          # y
         0,                             # dim
         0,                             # n_categories
         0                              # epoch
    )
end

"""
    train(art::SFAM, x, y)

Trains a Simple Fuzzy ARTMAP learner in a supervised manner.

# Examples
```julia-repl
julia> x, y = load_data()
julia> art = SFAM()
SFAM
    opts: opts_SFAM
    ...
julia> train!(art, x, y)
```
"""
function train!(art::SFAM, x::Array, y::Array ; preprocessed=false)
    # Get the correct dimensionality and number of samples
    if ndims(x) > 1
        art.dim, n_samples = size(x)
    else
        art.dim = 1
        n_samples = length(x)
    end

    # Initialize the internal categories
    art.y = zeros(Int, n_samples)

    # If the data is not preprocessed, then complement code it
    if !preprocessed
        x = complement_code(x)
    end

    # Initialize the training loop, continue to convergence
    art.epoch = 0
    while true
        art.epoch += 1
        iter = ProgressBar(1:n_samples)
        for ix in iter
            set_description(iter, string(@sprintf("Ep: %i, ID: %i, Cat: %i", art.epoch, ix, art.n_categories)))
            if !(y[ix] in art.labels)
                # Initialize W and labels
                if isempty(art.W)
                    art.W = Array{Float64}(undef, art.dim*2, 1)
                    art.W_old = Array{Float64}(undef, art.dim*2, 1)
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
end

"""
    classify(art::SFAM, x)

Categorize data 'x' using a trained Simple Fuzzy ARTMAP module 'art'.

# Examples
```julia-repl
julia> x, y = load_data()
julia> x_test, y_test = load_test_data()
julia> art = SFAM()
SFAM
    opts: opts_SFAM
    ...
julia> train!(art, x, y)
julia> classify(art, x_test)
```
"""
function classify(art::SFAM, x::Array ; preprocessed=false)
    # Get the correct dimensionality and number of samples
    if ndims(x) > 1
        art.dim, n_samples = size(x)
    else
        art.dim = 1
        n_samples = length(x)
    end
    y_hat = zeros(Int, n_samples)
    if !preprocessed
        x = complement_code(x)
    end

    iter = ProgressBar(1:n_samples)
    for ix in iter
        set_description(iter, string(@sprintf("ID: %i, Cat: %i", ix, art.n_categories)))

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
end

"""
    stopping_conditions(art::SFAM)

Stopping conditions for Simple Fuzzy ARTMAP, checked at the end of every epoch.
"""
function stopping_conditions(art::SFAM)
    # Compute the stopping condition, return a bool
    return art.W == art.W_old || art.epoch >= art.opts.max_epochs
end

"""
    learn(art::SFAM, x, W)

Returns a single updated weight for the Simple Fuzzy ARTMAP module for weight
vector W and sample x.
"""
function learn(art::SFAM, x::Array, W::Array)
    # Update W
    return art.opts.beta .* element_min(x, W) .+ W .* (1 - art.opts.beta)
end

"""
    activation(art::SFAM, x, W)

Returns the activation value of the Simple Fuzzy ARTMAP module with weight W
and sample x.
"""
function activation(art::SFAM, x::Array, W::Array)
    # Compute T and return
    return norm(element_min(x, W), 1) / (art.opts.alpha + norm(W, 1))
end

"""
    art_match(art::SFAM, x, W)

Returns the match function for the Simple Fuzzy ARTMAP module with weight W and
sample x.
"""
function art_match(art::SFAM, x::Array, W::Array)
    # Compute M and return
    return norm(element_min(x, W), 1) / art.dim
end
