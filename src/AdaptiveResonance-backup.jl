module ARTMAP-dev

using Logging
using Parameters
using Statistics
using LinearAlgebra

# using CUDAapi
# if has_cuda()		# Check if CUDA is available
#     @info "CUDA is on"
#     import CuArrays		# If CUDA is available, import CuArrays
#     CuArrays.allowscalar(false)
# end
include("funcs.jl")

export DDVFA, opts_DDVFA, GNFA, opts_GNFA, train!

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
    # @info "Successfully initialized opts_DDVFA"
end

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
    # "Reference" gamma for normalization: 0 <= gamma_ref < gamma
    gamma_ref = 1; @assert 0 <= gamma_ref && gamma_ref < gamma
    # Similarity method (activation and match):
    #   'single', 'average', 'complete', 'median', 'weighted', or 'centroid'
    method::String = "single"
    # Display flag
    display::Bool = true
    # shuffle::Bool = false
    random_seed = 1234.5678
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
    # @debug "Initializing Gamma-Normalized Fuzzy ART"
    # @debug "Successfully initialized Gamma-Normalized Fuzzy ART"
    opts = opts_GNFA()

    # Assign numerical parameters from options
    rho
    alpha
    beta
    gamma
    gamma_ref

    # Flag parameters
    method::String
    display::Bool = opts.display

    # Internal flags
    complement_coding::Bool = false
    max_epoch::Bool = false
    no_weight_change::Bool = false

    # Working variables
    threshold::Float64 = 0
    F2::Array{GNFA, 1} = []
    labels::Array{Int64, 1} = []
    T::Array{Float64, 1} = []
    M::Array{Float64, 1} = []
    n_samples::Int64 = 0
    n_categories::Int64 = 0
    dim::Int64 = 0
    dim_comp::Int64 = 0
    epoch::Int64 = 0

    # "Private" working variables
    sample::Array{Float64, 1} = []   # Current sample presented to DDVFA
    # W::Array{Float64, 1} = []        # All F2 nodes' weight vectors
    # W_old::Array{Float64, 1} = []    # Old F2 node weight vectors (for stopping criterion)
    W = []      # All F2 nodes' weight vectors
    W_old = []    # Old F2 node weight vectors (for stopping criterion)

end # GNFA
# @with_kw mutable struct GNFA
#     # @debug "Initializing Gamma-Normalized Fuzzy ART"
#     # @debug "Successfully initialized Gamma-Normalized Fuzzy ART"
#     opts = opts_GNFA()

#     # Assign numerical parameters from options
#     rho = opts.rho
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
#     T::Array{Float64, 1} = []
#     M::Array{Float64, 1} = []
#     n_samples::Int64 = 0
#     n_categories::Int64 = 0
#     dim::Int64 = 0
#     dim_comp::Int64 = 0
#     epoch::Int64 = 0

#     # "Private" working variables
#     sample::Array{Float64, 1} = []   # Current sample presented to DDVFA
#     # W::Array{Float64, 1} = []        # All F2 nodes' weight vectors
#     # W_old::Array{Float64, 1} = []    # Old F2 node weight vectors (for stopping criterion)
#     W = []      # All F2 nodes' weight vectors
#     W_old = []    # Old F2 node weight vectors (for stopping criterion)

# end # GNFA

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
@with_kw mutable struct DDVFA
    # @debug "Initializing DDVFA"
    # Get parameters
    opts = opts_DDVFA()
    # opts::opts_DDVFA
    subopts = opts_GNFA(rho=opts.rho_ub)

    # Assign numerical parameters from options
    rho = opts.rho_lb
    alpha = opts.alpha
    beta = opts.beta
    gamma = opts.gamma
    gamma_ref = opts.gamma_ref

    # Flag parameters
    method::String = opts.method
    display::Bool = opts.display

    # Internal flags
    complement_coding::Bool = false
    max_epoch::Bool = false
    no_weight_change::Bool = false

    # Working variables
    threshold::Float64 = 0
    F2::Array{GNFA, 1} = []
    labels::Array{Int64, 1} = []
    n_samples::Int64 = 0
    n_categories::Int64 = 0
    dim::Int64 = 0
    dim_comp::Int64 = 0
    epoch::Int64 = 0

    # "Private" working variables
    # sample::Array{Float64, 2} = []   # Current sample presented to DDVFA
    # W::Array{Float64, 2} = []       # All F2 nodes' weight vectors
    # W_old::Array{Float64, 2} = []   # Old F2 node weight vectors (for stopping criterion)
    W = []       # All F2 nodes' weight vectors
    W_old = []   # Old F2 node weight vectors (for stopping criterion)
    # Constructor that allows for empty working qs
    # DDVFA() = new()
    # @info "Successfully initialized DDVFA"
end # DDVFA

# DDVFA() where{T<:Real} = DDVFA{T}()

function train!(art::GNFA, data::Array, n_epochs::Int)
    # Initialization if empty
    if isempty(art.W)
        @info "Initializing GNFA"
        art.n_samples = 1
        art.n_categories = 1
        n_samples, art.dim_comp = size(data)
        art.dim = art.dim_comp/2 # Assumes input is already complement coded
        art.threshold = art.rho * (art.dim^art.gamma_ref)
        initial_sample = 2
        art.W = reshape(data[1, :], (1, art.dim_comp))
        # art.W = data[1, :]
    else
        initial_sample = 1
    end

    art.W_old = art.W

    # Learning
    art.epoch = 0
    while true
        art.epoch = art.epoch + 1
        # Loop over samples
        for i = initial_sample:n_samples
            # Check for already computed activation/match values
            if isempty(art.T) || isempty(art.M)
                # Compute activation/match functions
                activation_match!(art, data[i, :])
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
                    learn!(art, data[i, :], bmu)
                    # Update sample labels
                    art.labels[i] = bmu
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
                # art.W[art.n_categories, :] = data[i, :]
                # insert!(art.W, )
                art.W = [art.W; data[i,:]']
                # Increment number of samples associated with new category
                art.n_samples[art.n_categories, 1] = 1
                # Update sample labels
                art.label[i] = art.n_categories
            end
            # Empty activation vector
            art.T = []
            # Empty match vector
            art.M = []
            if art.display
                @info "Epoch: $art.epoch, Sample ID: $i, Categories: $art.n_categories"
            end
        end
    end
end

function activation_match!(art::GNFA, data::Array)
    art.T = zeros(art.n_categories)
    art.M = zeros(art.n_categories)
    for i = 1:art.n_categories
        W_norm = norm(art.W[i,:], 1)
        art.T[i, 1] = (norm(min(data, art.W[i, :]), 1)/(art.alpha + W_norm))^art.gamma
        art.M[i, 1] = (W_norm^art.gamma_ref)*art.T[i, 1]
    end
end

function learn!(art::GNFA, feature::Array, index::Int)
    art.W[index, :] = art.beta*(min(feature, art.W[index, :])) + (1-art.beta)*art.W[index, :]
    art.n[index, 1] = art.n_samples[index, 1] + 1
end

function stopping_conditions(art::GNFA, max_epochs::Int)
    stop = false
    if isequal(art.W, art.W_old)
        stop = true
    elseif art.epoch >= max_epochs
        stop = true
    end
    return stop
end

"""
    train(ddvfa, data, n_epochs)

Train the DDVFA model on the data for n_epochs
"""
function train!(art::DDVFA, data::Array, n_epochs::Int)
    @info "Training DDVFA"
    @info "DDVFA opts are" art.opts

    # Data information
    n_samples, dim = size(data)
    art.labels = zeros(n_samples)

    x = complement_coder(data)

end # train DDVFA

"""
    complement_coder(data)

Normalize the data x to [0, 1] and returns the augmented vector [x, 1 - x].
"""
function complement_coder(data)
    data_min = minimum(data)
    range = maximum(data) - data_min
    data_norm = (data .- data_min) ./ range
    complement = hcat(data_norm, 1 .- data_norm)
    return complement
end

"""
    get_field_meta(obj, field_name)

Get the value of a struct's field using meta programming.
"""
function get_field_meta(obj::Any, nafield_nameme::String)
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
function similarity_meta(method::String, F2, field_name::String, gamma_ref::AbstractFloat)
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
end # similarity

"""
    similarity(method, F2, field_name, gamma_ref)

Compute the similarity metric depending on method with explicit comparisons
for the field name.
"""
function similarity(method::String, F2, field_name::String, sample, gamma_ref::AbstractFloat)
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


end # module