# module DDVFA

using Logging
using Parameters
using Statistics
using LinearAlgebra
using ProgressBars
using Printf

# export DDVFA, opts_DDVFA, GNFA, opts_GNFA, train!

# struct DDVFA_hard
#    # Assign numerical parameters from options
#    rho
#    alpha
#    beta
#    gamma
#    gamma_ref

#    # Flag parameters
#    method::String
#    display::Bool

#    # Internal flags
#    complement_coding::Bool = false
#    max_epoch::Bool = false
#    no_weight_change::Bool = false

#    # Working variables
#    threshold::Float64 =
#    F2::Array{GNFA, 1}
#    labels::Array{Int64, 1}
#    n_samples::UInt128
#    dim::UInt128
#    epochs::UInt128 = 0

#    # "Private" working variables
#    sample::Array{Float64, 1}   # Current sample presented to DDVFA
#    W::Array{Float64, 1}        # All F2 nodes' weight vectors
#    W_old::Array{Float64, 1}
# end

"""
    opts_GNFA()

    Gamma-Normalized Fuzzy ART options struct.

    # Examples
    ```julia-repl
    julia> opts_GNFA()
    Initialized GNFA
    ```
"""
@with_kw mutable struct opts_GNFA @deftype Float64
    # @debug "Initializing opts_GNFA"
    # Vigilance parameter: [0, 1]
    rho = 0.6; @assert rho >= 0 && rho <= 1
    # Choice parameter: alpha > 0
    alpha = 1e-3; @assert alpha > 0
    # Learning parameter: (0, 1]
    beta = 1; @assert beta > 0 && beta <= 1
    # "Pseudo" kernel width: gamma >= 1
    gamma = 3; @assert gamma >= 1
    # gamma = 784; @assert gamma >= 1
    # "Reference" gamma for normalization: 0 <= gamma_ref < gamma
    gamma_ref = 1; @assert 0 <= gamma_ref && gamma_ref < gamma
    # Similarity method (activation and match):
    #   'single', 'average', 'complete', 'median', 'weighted', or 'centroid'
    method::String = "single"
    # Display flag
    display::Bool = true
    # shuffle::Bool = false
    random_seed = 1234.5678
    max_epochs = 1
    # @info "Successfully initialized opts_GNFA"
end # opts_GNFA

"""
    GNFA()

    Implements a GNFA learner.

    # Examples
    ```julia-repl
    julia> GNFA()
    GNFA
        opts: opts_GNFA
        ...
    ```
"""
mutable struct GNFA
    # Assign numerical parameters from options
    opts::opts_GNFA

    # Internal flags
    # complement_coding::Bool
    # max_epoch::Bool
    # no_weight_change::Bool

    # Working variables
    threshold::Float64
    labels::Array{Int, 1}
    T::Array{Float64, 1}
    M::Array{Float64, 1}
    # "Private" working variables
    W::Array{Float64, 2}
    W_old::Array{Float64, 2}
    n_instance::Array{Int, 1}
    n_categories::Int
    dim::Int
    dim_comp::Int
    epoch::Int
end # GNFA

function GNFA()
    # Get opts
    opts = opts_GNFA()

    GNFA(opts,                          # opts
        #  false,                         # complement_coding
        #  false,                         # max_epoch
        #  false,                         # no_weight_change
         0,                             # threshold
         Array{Int}(undef,0),           # labels
         Array{Float64}(undef, 0),      # T
         Array{Float64}(undef, 0),      # M
         Array{Float64}(undef, 0, 0),   # W
         Array{Float64}(undef, 0, 0),   # W_old
         Array{Int}(undef, 0),          # n_instance
         0,                             # n_categories
         0,                             # dim
         0,                             # dim_comp
         0                              # epoch
    )
end # GNFA()

function GNFA(opts)
    GNFA(opts,                          # opts
        #  false,                         # complement_coding
        #  false,                         # max_epoch
        #  false,                         # no_weight_change
         0,                             # threshold
         Array{Int}(undef,0),           # labels
         Array{Float64}(undef, 0),      # T
         Array{Float64}(undef, 0),      # M
         Array{Float64}(undef, 0, 0),   # W
         Array{Float64}(undef, 0, 0),   # W_old
         Array{Int}(undef, 0),          # n_instance
         0,                             # n_categories
         0,                             # dim
         0,                             # dim_comp
         0                              # epoch
    )
end # GNFA()

function initialize!(art::GNFA, x::Array)
    # @info "Initializing GNFA"
    art.dim_comp = size(x)[1]
    art.n_instance = [1]
    art.n_categories = 1
    art.dim = art.dim_comp/2 # Assumes input is already complement coded
    art.threshold = art.opts.rho * (art.dim^art.opts.gamma_ref)
    # initial_sample = 2
    art.W = Array{Float64}(undef, art.dim_comp, 1)
    # art.W[:, 1] = x[:, 1]
    art.W[:, 1] = x
    # label = supervised ? y[1] : 1
    # push!(art.labels, label)
end # initialize! GNFA

function train!(art::GNFA, x::Array ; y::Array=[])
    # Get size and if supervised
    if length(size(x)) == 2
        art.dim_comp, n_samples = size(x)
        prog_bar = true
    else
        art.dim_comp = length(x)
        n_samples = 1
        prog_bar = false
    end

    supervised = !isempty(y)
    # Initialization if empty
    if isempty(art.W)
        # @info "Initializing GNFA"
        # art.n_instance = [1]
        # art.n_categories = 1
        # art.dim = art.dim_comp/2 # Assumes input is already complement coded
        # art.threshold = art.opts.rho * (art.dim^art.opts.gamma_ref)
        # initial_sample = 2
        # art.W = Array{Float64}(undef, art.dim_comp, 1)
        # art.W[:, 1] = x[:, 1]
        label = supervised ? y[1] : 1
        push!(art.labels, label)
        initialize!(art, x[:, 1])
        initial_sample = 2
    else
        initial_sample = 1
    end

    art.W_old = deepcopy(art.W)

    # Learning
    art.epoch = 0
    while true
        art.epoch = art.epoch + 1
        # Loop over samples
        iter = prog_bar ? iter = ProgressBar(initial_sample:n_samples) : 1
        for i = iter
            if prog_bar
                set_description(iter, string(@sprintf("Ep: %i, ID: %i, Cat: %i", art.epoch, i, art.n_categories)))
            end
            # Check for already computed activation/match values
            if isempty(art.T) || isempty(art.M)
                # Compute activation/match functions
                activation_match!(art, x[:, i])
            end
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
                    learn!(art, x[:, i], bmu)
                    # Update sample labels
                    # art.labels[i] = bmu
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
                art.W = hcat(art.W, x[:,i])
                # Increment number of samples associated with new category
                # art.n_instance[art.n_categories] = 1
                push!(art.n_instance, 1)
                # Update sample labels
                # art.labels[i] = art.n_categories
                label = supervised ? y[i] : art.n_categories
                push!(art.labels, label)
            end
            # Empty activation and match vector
            art.T = []
            art.M = []
        end
        # Start from the first index from now on
        initial_sample = 1
        # Check for the stopping condition for the whole loop
        if stopping_conditions(art)
            break
        end
    end
end # train! GNFA

function classify(art::GNFA, x::Array)
    dim, n_samples = size(x)
    y_hat = zeros(Int, n_samples)
    # x = complement_code(x)

    iter = ProgressBar(1:n_samples)
    for ix in iter
        set_description(iter, string(@sprintf("ID: %i, Cat: %i", ix, art.n_categories)))
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
end # classify GNFA

# function element_min(x::Array, W::Array)
#     # Compute the element-wise minimum of two vectors
#     return minimum([x W], dims = 2)
# end # element_min

function activation_match!(art::GNFA, x::Array)
    art.T = zeros(art.n_categories)
    art.M = zeros(art.n_categories)
    for i = 1:art.n_categories
        W_norm = norm(art.W[:, i], 1)
        art.T[i] = (norm(element_min(x, art.W[:, i]), 1)/(art.opts.alpha + W_norm))^art.opts.gamma
        art.M[i] = (W_norm^art.opts.gamma_ref)*art.T[i]
    end
end # activation_match!

# Generic learning function
function learn(art::GNFA, x, W)
    # Update W
    return art.opts.beta .* element_min(x, W) .+ W .* (1 - art.opts.beta)
end
# In place learning function with instance counting
function learn!(art::GNFA, x, index)
    # Update W
    art.W[:, index] = learn(art, x, art.W[:, index])
    art.n_instance[index] += 1
end


function stopping_conditions(art::GNFA)
    # stop = false
    # if isequal(art.W, art.W_old) || art.epoch >= art.opts.max_epochs
    #     stop = true
    # end
    return isequal(art.W, art.W_old) || art.epoch >= art.opts.max_epochs
    # return stop
end # stopping_conditions GNFA






"""
    DDVFA()

    Implements a DDVFA learner.

    # Examples
    ```julia-repl
    julia> DDVFA()
    DDVFA
        opts: opts_DDVFA
        ...
    ```
"""
# @with_kw mutable struct DDVFA
#     # @debug "Initializing DDVFA"
#     # Get parameters
#     opts = opts_DDVFA()
#     # opts::opts_DDVFA
#     subopts = opts_GNFA(rho=opts.rho_ub)

#     # Assign numerical parameters from options
#     rho = opts.rho_lb
#     alpha = opts.alpha
#     beta = opts.beta
#     gamma = opts.gamma
#     gamma_ref = opts.gamma_ref

#     # Flag parameters
#     method::String = opts.method
#     display::Bool = opts.display

#     # Internal flags
#     complement_coding::Bool = false
#     max_epoch::Bool = false
#     no_weight_change::Bool = false

#     # Working variables
#     threshold::Float64 = 0
#     F2::Array{GNFA, 1} = []
#     labels::Array{Int64, 1} = []
#     n_samples::Int64 = 0
#     n_categories::Int64 = 0
#     dim::Int64 = 0
#     dim_comp::Int64 = 0
#     epoch::Int64 = 0

#     # "Private" working variables
#     # sample::Array{Float64, 2} = []   # Current sample presented to DDVFA
#     # W::Array{Float64, 2} = []       # All F2 nodes' weight vectors
#     # W_old::Array{Float64, 2} = []   # Old F2 node weight vectors (for stopping criterion)
#     W = []       # All F2 nodes' weight vectors
#     W_old = []   # Old F2 node weight vectors (for stopping criterion)
#     # Constructor that allows for empty working qs
#     # DDVFA() = new()
#     # @info "Successfully initialized DDVFA"
# end # DDVFA
# DDVFA() where{T<:Real} = DDVFA{T}()

"""
    opts_DDVFA()

    Distributed Dual Vigilance Fuzzy ART options struct.

    # Examples
    ```julia-repl
    julia> opts_DDVFA()
    Initialized opts_DDVFA
    ```
"""
@with_kw mutable struct opts_DDVFA @deftype Float64
    # @debug "Instantiating opts_DDVFA"
    # Lower-bound vigilance parameter: [0, 1]
    rho_lb = 0.80; @assert rho_lb >= 0 && rho_lb <= 1
    rho = rho_lb
    # Upper bound vigilance parameter: [0, 1]
    rho_ub = 0.85; @assert rho_ub >= 0 && rho_ub <= 1
    # Choice parameter: alpha > 0
    alpha = 1e-3; @assert alpha > 0
    # Learning parameter: (0, 1]
    beta = 1; @assert beta > 0 && beta <= 1
    # "Pseudo" kernel width: gamma >= 1
    gamma = 3; @assert gamma >= 1
    # "Reference" gamma for normalization: 0 <= gamma_ref < gamma
    gamma_ref = 1; @assert 0 <= gamma_ref && gamma_ref < gamma
    # Similarity method (activation and match):
    #   'single', 'average', 'complete', 'median', 'weighted', or 'centroid'
    method::String = "single"
    # Display flag
    display::Bool = true
    # shuffle::Bool = false
    random_seed = 1234.5678
    max_epoch = 1
end # opts_DDVFA


mutable struct DDVFA
    # Get parameters
    opts::opts_DDVFA
    subopts::opts_GNFA

    # Working variables
    threshold::Float64
    F2::Array{GNFA, 1}
    labels::Array{Int, 1}
    W::Array{Float64, 2}        # All F2 nodes' weight vectors
    W_old::Array{Float64, 2}    # Old F2 node weight vectors (for stopping criterion)
    # n_samples::Int
    n_categories::Int
    dim::Int
    dim_comp::Int
    epoch::Int
end # DDVFA

function DDVFA()
    opts = opts_DDVFA()
    subopts = opts_GNFA(rho=opts.rho_ub)
    DDVFA(opts,
          subopts,
          0,
          Array{GNFA}(undef, 0),
          Array{Int}(undef, 0),
          Array{Float64}(undef, 0, 0),
          Array{Float64}(undef, 0, 0),
        #   0,
          0,
          0,
          0,
          0
    )

end # DDVFA()

"""
    train(ddvfa, data, n_epochs)

Train the DDVFA model on the data for n_epochs
"""
function train!(art::DDVFA, x::Array)
    @info "Training DDVFA"

    # Data information
    art.dim, n_samples = size(x)
    art.dim_comp = 2*art.dim
    art.labels = zeros(n_samples)

    x = complement_code(x)

    # Initialization
    if isempty(art.F2)
        # Global Fuzzy ART
        art.n_categories = 1
        art.labels[1] = 1
        # Local Fuzzy ART
        # art.F2[art.n_categories] = GNFA(art.subopts)
        push!(art.F2, GNFA(art.subopts))
        initialize!(art.F2[1], x[:, 1])
        initial_sample = 2
    else
        initial_sample = 1
    end

    # art.W_old = deepcopy(art.F2[])
    art.W_old = Array{Float64}(undef, art.dim_comp, 1)
    art.W_old[:, 1] = x[:, 1]

    # Learning
    art.threshold = art.opts.rho*(art.dim^art.opts.gamma_ref)
    art.epoch = 0
    while true
        art.epoch += 1
        iter = ProgressBar(initial_sample:n_samples)
        for i = iter
            set_description(iter, string(@sprintf("Ep: %i, ID: %i, Cat: %i", art.epoch, i, art.n_categories)))
            sample = x[:, i]
            T = zeros(art.n_categories)
            for jx = 1:art.n_categories
                activation_match!(art.F2[jx], sample)
                T[jx] = similarity(art.opts.method, art.F2[jx], "T", sample, art.opts.gamma_ref)
            end
            index = sortperm(T, rev=true)
            mismatch_flag = true
            for jx = 1:art.n_categories
                bmu = index[jx]
                M = similarity(art.opts.method, art.F2[bmu], "M", sample, art.opts.gamma_ref)
                if M >= art.threshold
                    # # DIAGNOSTIC
                    # if i == 125
                    #     @info "Sample", sample, "BMU", bmu, "M", M, "index", index[1:5]
                    # end
                    train!(art.F2[bmu], sample)
                    art.labels[i] = bmu
                    mismatch_flag = false
                    break
                end
            end
            if mismatch_flag
                # Global Fuzzy ART
                art.n_categories += 1
                push!(art.labels, art.n_categories)
                # Local Fuzzy ART
                push!(art.F2, GNFA(art.subopts))
                initialize!(art.F2[art.n_categories], sample)
            end
        end
        # Make sure to start at first sample from now on
        initial_sample = 1
        # art.W = []
        # art.W = Array{Float64}(undef, art.dim*2, 1)
        art.W = art.F2[1].W
        for kx = 2:art.n_categories
            art.W = [art.W art.F2[kx].W]
        end
        if stopping_conditions(art)
            break
        end
        art.W_old = deepcopy(art.W)
    end
end # train DDVFA

function stopping_conditions(art::DDVFA)
    return art.W == art.W_old || art.epoch >= art.opts.max_epoch
end # stopping_conditions

# """
#     complement_code(data)

# Normalize the data x to [0, 1] and returns the augmented vector [x, 1 - x].
# """
# function complement_code(data::Array)
#     # Complement code the data and return a concatenated matrix
#     dim, n_samples = size(data)
#     x_raw = zeros(dim, n_samples)

#     mins = [minimum(data[i, :]) for i in 1:dim]
#     maxs = [maximum(data[i, :]) for i in 1:dim]

#     for i = 1:dim
#         if maxs[i] - mins[i] != 0
#             x_raw[i, :] = (data[i, :] .- mins[i]) ./ (maxs[i] - mins[i])
#         end
#     end

#     x = vcat(x_raw, 1 .- x_raw)
#     return x
# end


"""
    get_field_meta(obj, field_name)

Get the value of a struct's field using meta programming.
"""
function get_field_meta(obj::Any, field_name::String)
    field = Symbol(field_name)
    code = quote
        (obj) -> obj.$field
    end
    return eval(code)
end

"""
    get_field_native(obj, field_name)

Get the value of a struct's field through the julia native method.
"""
function get_field_native(obj::Any, field_name::String)
    return getfield(obj, Symbol(field_name))
end

"""
    similarity_meta(method, F2, field_name, gamma_ref)

Compute the similarity metric depending on method using meta programming to
access the correct field.
"""
function similarity_meta(method::String, F2::GNFA, field_name::String, gamma_ref::AbstractFloat)
    @debug "Computing similarity"

    if field_name != "T" && field_name != "M"
        error("Incorrect field name for similarity metric.")
    end

    field = get_field_native(F2, field_name)

    # Single linkage
    if method == "single"
        value = maximum(field)
    # Average linkage
    elseif method == "average"
        value = mean(field)
    # Complete linkage
    elseif method == "complete"
        value = minimum(field)
    # Median linkage
    elseif method == "median"
        value = median(field)
    elseif method == "weighted"
        value = field' * (F2.n / sum(F2.n))
    elseif method == "centroid"
        Wc = minimum(F2.W)
        T = norm(min(sample, Wc), 1)
        if field_name == "T"
            value = T
        elseif field_name == "M"
            value = (norm(Wc, 1)^gamma_ref)*T
        end
    else
        error("Invalid/unimplemented similarity method")
    end
    return value
end # similarity_meta

"""
    similarity(method, F2, field_name, gamma_ref)

Compute the similarity metric depending on method with explicit comparisons
for the field name.
"""
function similarity(method::String, F2::GNFA, field_name::String, sample::Array, gamma_ref::AbstractFloat)
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
            value = F2.T * (F2.n / sum(F2.n))
        elseif field_name == "M"
            value = F2.M * (F2.n / sum(F2.n))
        end
    # Centroid linkage
    elseif method == "centroid"
        Wc = minimum(F2.W)
        T = norm(min(sample, Wc), 1)
        if field_name == "T"
            value = T
        elseif field_name == "M"
            value = (norm(Wc, 1)^gamma_ref)*T
        end
    else
        error("Invalid/unimplemented similarity method")
    end
end # similarity

# end