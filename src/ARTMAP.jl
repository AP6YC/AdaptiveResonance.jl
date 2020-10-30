using Parameters
using LinearAlgebra
using Logging
using ProgressBars
using Printf
using MLJ


"""
    opts_FAM()

Implements a Fuzzy ARTMAP learner's options.

# Examples
```julia-repl
julia> my_opts = opts_FAM()
```
"""
@with_kw mutable struct opts_FAM <: AbstractARTOpts @deftype Float64
    # Vigilance parameter: [0, 1]
    rho = 0.6; @assert rho >= 0 && rho <= 1
    # Choice parameter: alpha > 0
    alpha = 1e-7; @assert alpha > 0
    # Match tracking parameter
    epsilon = 1e-3; @assert epsilon > 0 && epsilon < 1
    # Learning parameter: (0, 1]
    beta = 1; @assert beta > 0 && beta <= 1
    # Similarity method (activation and match):
    #   'single', 'average', 'complete', 'median', 'weighted', or 'centroid'
    method::String = "single"
    # Uncommitted node flag
    uncommitted::Bool = true
    # Display flag
    display::Bool = true
    # shuffle::Bool = false
    random_seed = 1234.5678
    max_epochs = 1
end


"""
    FAM

Fuzzy ARTMAP struct.
"""
mutable struct FAM <: AbstractART
    opts::opts_FAM
    W::Array{Float64, 2}
    W_old::Array{Float64, 2}
    labels::Array{Int, 1}
    y::Array{Int, 1}
    dim::Int
    n_categories::Int
    epoch::Int
end


"""
    FAM()

Implements a Fuzzy ARTMAP learner.

# Examples
```julia-repl
julia> FAM()
FAM
    opts: opts_FAM
    ...
```
"""
function FAM()
    opts = opts_FAM()
    FAM(opts)
end


"""
    FAM(opts)

Implements a Fuzzy ARTMAP learner with specified options.

# Examples
```julia-repl
julia> opts = opts_FAM()
julia> FAM(opts)
FAM
    opts: opts_FAM
    ...
```
"""
function FAM(opts::opts_FAM)
    FAM(opts,
        Array{Float64}(undef, 0,0),
        Array{Float64}(undef, 0,0),
        Array{Int}(undef, 0),
        Array{Int}(undef, 0),
        0, 0, 0)
end


"""
    opts_DAM()

Implements a Default ARTMAP learner's options.

# Examples
```julia-repl
julia> my_opts = opts_DAM()
```
"""
@with_kw mutable struct opts_DAM <: AbstractARTOpts @deftype Float64
    # Vigilance parameter: [0, 1]
    rho = 0.6; @assert rho >= 0 && rho <= 1
    # Choice parameter: alpha > 0
    alpha = 1e-7; @assert alpha > 0
    # Match tracking parameter
    epsilon = -1e-3; @assert epsilon > -1 && epsilon < 1
    # Learning parameter: (0, 1]
    beta = 1; @assert beta > 0 && beta <= 1
    # Similarity method (activation and match):
    #   'single', 'average', 'complete', 'median', 'weighted', or 'centroid'
    # method::String = "single"
    # Display flag
    display::Bool = true
    # shuffle::Bool = false
    random_seed = 1234.5678
    max_epochs = 1
end


"""
    DAM

Default ARTMAP struct.
"""
mutable struct DAM <: AbstractART
    opts::opts_DAM
    W::Array{Float64, 2}
    W_old::Array{Float64, 2}
    labels::Array{Int, 1}
    y::Array{Int, 1}
    dim::Int
    n_categories::Int
    epoch::Int
end


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
end


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
    DAM(opts,
        Array{Float64}(undef, 0,0),
        Array{Float64}(undef, 0,0),
        Array{Int}(undef, 0),
        Array{Int}(undef, 0),
        0, 0, 0)
end


"""
    train(art::DAM, x, y)

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
function train!(art::DAM, x::Array, y::Array)
    art.dim, n_samples = size(x)
    art.y = zeros(Int, n_samples)
    x = complement_code(x)
    art.epoch = 0

    # Convenient semantic flag
    # is_supervised = !isempty(y)

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
    classify(art, x)

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
function classify(art::DAM, x::Array)
    art.dim, n_samples = size(x)
    y_hat = zeros(Int, n_samples)
    x = complement_code(x)

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
    stopping_conditions(art::DAM)

Stopping conditions for Default ARTMAP, checked at the end of every epoch.
"""
function stopping_conditions(art::DAM)
    # Compute the stopping condition, return a bool
    return art.W == art.W_old || art.epoch >= art.opts.max_epochs
end


"""
    activation(art::DAM, x, W)

Default ARTMAP's choice-by-difference activation function.
"""
function activation(art::DAM, x::Array, W::Array)
    # Compute T and return
    return norm(element_min(x, W), 1) +
        (1-art.opts.alpha)*(art.dim - norm(W, 1))
end


"""
    learn(art::DAM, x, W)

Returns a single updated weight for the Simple Fuzzy ARTMAP module for weight
vector W and sample x.
"""
function learn(art::DAM, x::Array, W::Array)
    # Update W
    return art.opts.beta .* element_min(x, W) .+ W .* (1 - art.opts.beta)
end


"""
    art_match(art::DAM, x, W)

Returns the match function for the Default ARTMAP module with weight W and
sample x.
"""
function art_match(art::DAM, x::Array, W::Array)
    # Compute M and return
    return norm(element_min(x, W), 1) / art.dim
end


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
    # Similarity method (activation and match):
    #   'single', 'average', 'complete', 'median', 'weighted', or 'centroid'
    method::String = "single"
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
    SFAM(opts,
         Array{Float64}(undef, 0,0),
         Array{Float64}(undef, 0,0),
         Array{Int}(undef, 0),
         Array{Int}(undef, 0),
         0, 0, 0)
end


"""
    train(art, x, y)

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
function train!(art::SFAM, x::Array, y::Array)
    art.dim, n_samples = size(x)
    art.y = zeros(Int, n_samples)
    x = complement_code(x)
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
    classify(art, x)

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
function classify(art::SFAM, x::Array)
    art.dim, n_samples = size(x)
    y_hat = zeros(Int, n_samples)
    x = complement_code(x)

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


"""
    performance(y_hat, y)

Returns the categorization performance of y_hat against y.
"""
function performance(y_hat::Array, y::Array)
    # Compute the confusion matrix and calculate performance as trace/sum
    conf = confusion_matrix(categorical(y_hat), categorical(y), warn=false)
    return tr(conf.mat)/sum(conf.mat)
end
