"""
    opts_FuzzyART()

Implements a Fuzzy ART learner's options.

# Examples
```julia-repl
julia> my_opts = opts_FuzzyART()
```
"""
@with_kw mutable struct opts_FuzzyART <: AbstractARTOpts @deftype Float64
    # Vigilance parameter: [0, 1]
    rho = 0.5; @assert rho >= 0 && rho <= 1
    # Choice parameter: alpha > 0
    alpha = 1e-3; @assert alpha > 0
    # Learning parameter: (0, 1]
    beta = 1; @assert beta > 0 && beta <= 1
    # Uncommitted node flag
    uncommitted::Bool = true
    # Display flag
    display::Bool = true
    # Maximum numbers of epochs to train for
    max_epochs = 1
    # Flag to find second best matching unit
    find_bmu2 = false
end # opts_FuzzyART()

"""
    FuzzyART

Simple Fuzzy ART struct.
"""
mutable struct FuzzyART <: AbstractART
    opts::opts_FuzzyART         # Fuzzy ART module options
    config::DataConfig          # data configuration
    W::Array{Float64, 2}        # top-down weights
    W_old::Array{Float64, 2}    # old top-down weight values
    labels::Array{Int, 1}       # best matching units (class labels)
    labels2::Array{Int, 1}      # second best matching units
    n_categories::Int           # total number of categories
    n_instance::Array{Int, 1}   # instance counting
    epoch::Int                  # current epoch
end # FuzzyART <: AbstractART

"""
    FuzzyART()

Implements a Fuzzy ART learner.

# Examples
```julia-repl
julia> FuzzyART()
FuzzyART
    opts: opts_FuzzyART
    ...
```
"""
function FuzzyART()
    opts = opts_FuzzyART()
    FuzzyART(opts)
end # FuzzyART()

"""
    FuzzyART(opts)

Implements a Simple Fuzzy ARTMAP learner with specified options.

# Examples
```julia-repl
julia> opts = opts_FuzzyART()
julia> FuzzyART(opts)
FuzzyART
    opts: opts_FuzzyART
    ...
```
"""
function FuzzyART(opts::opts_FuzzyART)
    FuzzyART(
        opts,                           # opts_FuzzyART
        DataConfig(),                   # config
        Array{Float64}(undef, 0,0),     # W
        Array{Float64}(undef, 0,0),     # W_old
        Array{Int}(undef, 0),           # labels
        Array{Int}(undef, 0),           # labels2
        0,                              # n_categories
        Array{Int}(undef, 0),           # n_instance
        0                               # epoch
    )
end # FuzzyART(opts::opts_FuzzyART)

"""
    train!(art::FuzzyART, x::Array ; preprocessed=false)

Trains a Fuzzy ART learner.

# Examples
```julia-repl
julia> x, y = load_data()
julia> art = FuzzyART()
FuzzyART
    opts: opts_FuzzyART
    ...
julia> train!(art, x, y)
```
"""
function train!(art::FuzzyART, x::Array ; preprocessed=false)
    # Show a message if display is on
    art.opts.display && @info "Training FuzzyART"

    # Data information and setup
    n_samples = get_n_samples(x)

    # Set up the data config if it is not already
    !art.config.setup && data_setup!(art.config, x)

    # If the data is not preprocessed, then complement code it
    if !preprocessed
        x = complement_code(x, config=art.config)
    end

    # Initialization
    if isempty(art.W)
        art.W = ones(Float64, art.config.dim_comp)
        push!(art.n_instance, 0)
        art.n_categories = 1
    end

    # Initialize the internal categories
    art.labels = zeros(Int, n_samples)
    art.labels2 = zeros(Int, n_samples)
    art.W_old = deepcopy(art.W)

    # Initialize the training loop, continue to convergence
    art.epoch = 0
    while true
        art.epoch += 1
        iter = get_iterator(art.opts, x)
        for ix in iter
            # Update the iterator if necessary
            update_iter(art, iter, ix)
            # Grab the sample
            sample = get_sample(x, ix)
            # Compute the activation and match
            activation_match!(art, sample)

            # Sort activation function values in descending order
            index = sortperm(T, rev=true)
            # Mismatch for bmu and second bmu default to true
            mismatch_flag = true
            mismatch_flag_2 = true

            for jx = 1:art.n_categories
                bmu = index[jx]
                if art.M[bmu] >= art.opts.rho
                    # if obj.vc2(bmu)
                    if mismatch_flag
                        learn!(art, sample, bmu)
                        art.labels[bmu] = bmu
                        mismatch_flag = false
                        if !art.opts.find_bmu2
                            mismatch_flag_2 = false
                            break
                        end
                    else
                        cvi.labels2[i] = bmu
                        break
                    end
                end
            end
        end



    #         if !(y[ix] in art.labels)
    #             # Initialize W and labels
    #             if isempty(art.W)
    #                 art.W = Array{Float64}(undef, art.config.dim_comp, 1)
    #                 art.W_old = Array{Float64}(undef, art.config.dim_comp, 1)
    #                 art.W[:, ix] = x[:, ix]
    #             else
    #                 art.W = [art.W x[:, ix]]
    #             end
    #             push!(art.labels, y[ix])
    #             art.n_categories += 1
    #             art.y[ix] = y[ix]
    #         else
    #             # Baseline vigilance parameter
    #             rho_baseline = art.opts.rho

    #             # Compute activation function
    #             T = zeros(art.n_categories)
    #             for jx in 1:art.n_categories
    #                 T[jx] = activation(art, x[:, ix], art.W[:, jx])
    #             end

    #             # Sort activation function values in descending order
    #             index = sortperm(T, rev=true)
    #             mismatch_flag = true
    #             for jx in 1:art.n_categories
    #                 # Compute match function
    #                 M = art_match(art, x[:, ix], art.W[:, index[jx]])
    #                 # Current winner
    #                 if M >= rho_baseline
    #                     if y[ix] == art.labels[index[jx]]
    #                         # Learn
    #                         @debug "Learning"
    #                         art.W[:, index[jx]] = learn(art, x[:, ix], art.W[:, index[jx]])
    #                         art.y[ix] = art.labels[index[jx]]
    #                         mismatch_flag = false
    #                         break
    #                     else
    #                         # Match tracking
    #                         @debug "Match tracking"
    #                         rho_baseline = M + art.opts.epsilon
    #                     end
    #                 end
    #             end
    #             if mismatch_flag
    #                 # Create new weight vector
    #                 @debug "Mismatch"
    #                 art.W = hcat(art.W, x[:, ix])
    #                 push!(art.labels, y[ix])
    #                 art.n_categories += 1
    #                 art.y[ix] = y[ix]
    #             end
    #         end
    #     end
    #     if stopping_conditions(art)
    #         break
    #     end
    #     art.W_old = deepcopy(art.W)
    # end
end # train!(art::FuzzyART, x::Array ; preprocessed=false)

function resonance_check(art::FuzzyART, a::Int)
    b = art.labels[end]
    is_res = true
    nb = art.n_instance[b]
    if nb == 2
        is_res = false
    elseif a <= size(art.map, 2) && any(art.map[1:b-1])
end

"""
    classify(art::FuzzyART, x::Array ; preprocessed=false)

Categorize data 'x' using a trained Fuzzy ART module 'art'.

# Examples
```julia-repl
julia> x, y = load_data()
julia> x_test, y_test = load_test_data()
julia> art = FuzzyART()
FuzzyART
    opts: opts_FuzzyART
    ...
julia> train!(art, x, y)
julia> classify(art, x_test)
```
"""
function classify(art::FuzzyART, x::Array ; preprocessed=false)
    # Show a message if display is on
    art.opts.display && @info "Testing FuzzyART"

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
            # Current winner
            if M >= art.opts.rho
                y_hat[ix] = art.labels[index[jx]]
                mismatch_flag = false
                break
            end
        end
        if mismatch_flag
            # Label as -1 if mismatched
            @debug "Mismatch"
            y_hat[ix] = -1
        end
    end
    return y_hat
end # classify(art::FuzzyART, x::Array ; preprocessed=false)

"""
    stopping_conditions(art::FuzzyART)

Stopping conditions for Fuzzy ART, checked at the end of every epoch.
"""
function stopping_conditions(art::FuzzyART)
    # Compute the stopping condition, return a bool
    return art.W == art.W_old || art.epoch >= art.opts.max_epochs
end # stopping_conditions(art::FuzzyART)

# """
#     learn(art::FuzzyART, x::Array, W::Array)

# Returns a single updated weight for the Fuzzy ART module for weight
# vector W and sample x.
# """
# function learn(art::FuzzyART, x::Array, W::Array)
#     # Update W
#     return art.opts.beta .* element_min(x, W) .+ W .* (1 - art.opts.beta)
# end # learn(art::FuzzyART, x::Array, W::Array)

"""
    learn!(art::FuzzyART, x::Array, ix::Int)

Mutates the weight vector in place at index 'ix' by sample 'x', updating the instance counter.
"""
function learn!(art::FuzzyART, x::Array, ix::Int)
    art.W[:, ix] = art.opts.beta .* element_min(x, art.W[:, ix]) .+ art.W[:, ix] .* (1 - art.opts.beta)
    art.n_instance[ix] += 1
end # learn!(art::FuzzyART, x::Array, ix::Int)

# """
#     activation(art::FuzzyART, x::Array, W::Array)

# Returns the activation value of the Fuzzy ART module with weight W
# and sample x.
# """
# function activation(art::FuzzyART, x::Array, W::Array)
#     # Compute T and return
#     return norm(element_min(x, W), 1) / (art.opts.alpha + norm(W, 1))
# end # activation(art::FuzzyART, x::Array, W::Array)

# """
#     art_match(art::FuzzyART, x::Array, W::Array)

# Returns the match function for the Fuzzy ART module with weight W and
# sample x.
# """
# function art_match(art::FuzzyART, x::Array, W::Array)
#     # Compute M and return
#     return norm(element_min(x, W), 1) / art.config.dim
#     # return norm(element_min(x, W), 1) / art.config.dim_comp
# end # art_match(art::FuzzyART, x::Array, W::Array)

"""
    clear_activation_match!(art::FuzzyART)

Clear the Fuzzy ART activation and match vectors.
"""
function clear_activation_match!(art::FuzzyART)
    # Clear the activation and match
    art.T = zeros(art.n_categories)
    art.M = zeros(art.n_categories)
end # clear_activation_match!(art::FuzzyART)

"""
    activation_match!(art::FuzzyART, x::Array)

Compute the Fuzzy ART activation and match vectors.
"""
function activation_match!(art::FuzzyART, x::Array)
    clear_activation_match!(art)
    for ix = 1:art.n_categories
        numerator = norm(element_min(x, art.W[:, ix]), 1)
        art.T[ix] = numerator / (art.opts.alpha + norm(W[:, ix], 1))
        art.M[ix] = numerator / art.config.dim
    end
end # activation_match!(art::FuzzyART, x::Array)