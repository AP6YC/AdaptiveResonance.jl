"""
    opts_GNFA()

Gamma-Normalized Fuzzy ART options struct.

# Examples
```julia-repl
julia> opts_GNFA()
Initialized GNFA
```
"""
@with_kw mutable struct opts_GNFA <: AbstractARTOpts @deftype Float64
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

    max_epochs = 1
end # opts_GNFA

"""
    GNFA <: AbstractART

Gamma-Normalized Fuzzy ART learner struct

# Examples
```julia-repl
julia> GNFA()
GNFA
    opts: opts_GNFA
    ...
```
"""
mutable struct GNFA <: AbstractART
    # Assign numerical parameters from options
    opts::opts_GNFA
    config::DataConfig

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
end # GNFA(opts)

"""
    initialize!(art::GNFA, x::Array)

Initializes a GNFA learner with an intial sample 'x'

# Examples
```julia-repl
julia> my_GNFA = GNFA()
GNFA
    opts: opts_GNFA
    ...
julia> initialize!(my_GNFA, [1 2 3 4])
```
"""
function initialize!(art::GNFA, x::Array)
    # Set up the data config
    if art.config.setup
        @warn "Data configuration already set up, overwriting config"
    else
        art.config.setup = true
    end
    art.config.dim_comp = size(x)[1]
    art.config.dim = Int(art.config.dim_comp/2) # Assumes input is already complement coded

    art.n_instance = [1]
    art.n_categories = 1

    art.threshold = art.opts.rho * (art.config.dim^art.opts.gamma_ref)
    # initial_sample = 2
    art.W = Array{Float64}(undef, art.config.dim_comp, 1)
    # art.W[:, 1] = x[:, 1]
    art.W[:, 1] = x
    # label = supervised ? y[1] : 1
    # push!(art.labels, label)
end # initialize!(GNFA, x)

"""
    train!(art::GNFA, x::Array ; y::Array=[])

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
function train!(art::GNFA, x::Array ; y::Array=[])
    # Get the number of samples to process
    # n_samples = get_n_samples(x)

    # Show a progbar only if the data is 2-D and the option is on
    single_sample = length(size(x)) == 1
    prog_bar = single_sample ? false : art.opts.display
    n_samples = single_sample ? 1: size(x)[2]
    # prog_bar = length(size(x)) == 2 ? art.opts.display : false
    # n_samples = length(size(x)) == 2 ? size(x)[2] : 1
    # # Get size and if supervised
    # if length(size(x)) == 2
    #     art.dim_comp, n_samples = size(x)
    #     # Create a progressbar only if the display flag is set
    #     prog_bar = art.opts.display
    # else
    #     art.dim_comp = length(x)
    #     n_samples = 1
    #     # No progress bar even if display is set since learning a single sample
    #     prog_bar = false
    # end

    supervised = !isempty(y)

    # Initialization if weights are empty; fast commit the first sample
    if isempty(art.W)
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
        iter_raw = initial_sample:n_samples
        iter = prog_bar ?  ProgressBar(iter_raw) : iter_raw
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
end # train!(GNFA, x, y=[])

"""
    classify(art::GNFA, x::Array)

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
function classify(art::GNFA, x::Array)
    # Get the number of samples to classify
    n_samples = get_n_samples(x)

    # Initialize the output vector and iterate across all data
    y_hat = zeros(Int, n_samples)
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
end # classify(GNFA, x)

"""
    activation_match!(art::GNFA, x::Array)

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
function activation_match!(art::GNFA, x::Array)
    art.T = zeros(art.n_categories)
    art.M = zeros(art.n_categories)
    for i = 1:art.n_categories
        W_norm = norm(art.W[:, i], 1)
        art.T[i] = (norm(element_min(x, art.W[:, i]), 1)/(art.opts.alpha + W_norm))^art.opts.gamma
        art.M[i] = (W_norm^art.opts.gamma_ref)*art.T[i]
    end
end # activation_match!(GNFA, x)

"""
    learn(art::GNFA, x::Array, W::Array)

Return the modified weight of the art module conditioned by sample x.
"""
function learn(art::GNFA, x::Array, W::Array)
    # Update W
    return art.opts.beta .* element_min(x, W) .+ W .* (1 - art.opts.beta)
end # learn(art::GNFA, x::Array, W::Array)

"""
    learn!(art::GNFA, x::Array, index::Int)

In place learning function with instance counting.
"""
function learn!(art::GNFA, x::Array, index::Int)
    # Update W
    art.W[:, index] = learn(art, x, art.W[:, index])
    art.n_instance[index] += 1
end # learn!(art::GNFA, x::Array, index::Int)

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
julia> opts_DDVFA()
Initialized opts_DDVFA
```
"""
@with_kw mutable struct opts_DDVFA <: AbstractARTOpts @deftype Float64
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

    max_epoch = 1
end # opts_DDVFA

"""
    DDVFA <: AbstractART

Distributed Dual Vigilance Fuzzy ARTMAP module struct.

# Examples
```julia-repl
julia> DDVFA()
DDVFA
    opts: opts_DDVFA
    supopts::opts_GNFA
    ...
```
"""
mutable struct DDVFA <: AbstractART
    # Get parameters
    opts::opts_DDVFA
    subopts::opts_GNFA
    config::DataConfig

    # Working variables
    threshold::Float64
    F2::Array{GNFA, 1}
    labels::Array{Int, 1}
    W::Array{Float64, 2}        # All F2 nodes' weight vectors
    W_old::Array{Float64, 2}    # Old F2 node weight vectors (for stopping criterion)
    # n_samples::Int
    n_categories::Int
    epoch::Int
end # DDVFA

"""
    DDVFA()

Implements a DDVFA learner with default options.

# Examples
```julia-repl
julia> DDVFA()
DDVFA
    opts: opts_DDVFA
    supopts: opts_GNFA
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
    supopts: opts_GNFA
    ...
```
"""
function DDVFA(opts::opts_DDVFA)
    subopts = opts_GNFA(rho=opts.rho_ub)
    DDVFA(opts,
          subopts,
          DataConfig(),
          0,
          Array{GNFA}(undef, 0),
          Array{Int}(undef, 0),
          Array{Float64}(undef, 0, 0),
          Array{Float64}(undef, 0, 0),
          0,
          0
    )
end # DDVFA(opts)

"""
    train!(art::DDVFA, x::Array ; preprocessed=false)

Train the DDVFA model on the data.
"""
function train!(art::DDVFA, x::Array ; preprocessed=false)
    # Show a message if display is on
    art.opts.display && @info "Training DDVFA"

    # Data information and setup
    n_samples = get_n_samples(x)

    # Set up the data config if training for the first time
    !art.config.setup && data_setup!(art.config, x)

    # If the data is not preprocessed, then complement code it
    if !preprocessed
        x = complement_code(x, art.config)
    end

    art.labels = zeros(n_samples)

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
    art.W_old = Array{Float64}(undef, art.config.dim_comp, 1)
    art.W_old[:, 1] = x[:, 1]

    # Learning
    art.threshold = art.opts.rho*(art.config.dim^art.opts.gamma_ref)
    art.epoch = 0
    while true
        art.epoch += 1
        iter_raw = initial_sample:n_samples
        iter = art.opts.display ? ProgressBar(iter_raw) : iter_raw
        for i = iter
            if art.opts.display
                set_description(iter, string(@sprintf("Ep: %i, ID: %i, Cat: %i", art.epoch, i, art.n_categories)))
            end
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
        # art.W = Array{Float64}(undef, art.config.dim*2, 1)
        art.W = art.F2[1].W
        for kx = 2:art.n_categories
            art.W = [art.W art.F2[kx].W]
        end
        if stopping_conditions(art)
            break
        end
        art.W_old = deepcopy(art.W)
    end
end # train!(art::DDVFA, x::Array ; preprocessed=false)

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
    similarity(method::String, F2::GNFA, field_name::String, sample::Array, gamma_ref::Real)

Compute the similarity metric depending on method with explicit comparisons
for the field name.
"""
function similarity(method::String, F2::GNFA, field_name::String, sample::Array, gamma_ref::Real)
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
        Wc = minimum(F2.W, dims=2)
        # (norm(min(obj.sample, Wc), 1)/(obj.alpha + norm(Wc, 1)))^obj.gamma;
        T = norm(element_min(sample, Wc), 1) / (F2.opts.alpha + norm(Wc, 1))^F2.opts.gamma
        if field_name == "T"
            value = T
        elseif field_name == "M"
            value = (norm(Wc, 1)^gamma_ref)*T
        end
    else
        error("Invalid/unimplemented similarity method")
    end
end # similarity(method::String, F2::GNFA, field_name::String, sample::Array, gamma_ref::Real)

"""
    classify(art::DDVFA, x::Array ; preprocessed=false)

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
function classify(art::DDVFA, x::Array ; preprocessed=false)
    # Show a message if display is on
    art.opts.display && @info "Testing DDVFA"

    # Data information and setup
    n_samples = get_n_samples(x)

    if !art.config.setup
        @error "Attempting to classify data before setup"
    end

    if !preprocessed
        x = complement_code(x, art.config)
    end

    # Data information
    # art.dim, n_samples = size(x)
    # _, n_samples = size(x)
    # art.dim_comp = 2*art.dim
    # art.labels = zeros(n_samples)
    y_hat = zeros(Int, n_samples)

    iter_raw = 1:n_samples
    iter = art.opts.display ? ProgressBar(iter_raw) : iter_raw
    for ix = iter
        if art.opts.display
            set_description(iter, string(@sprintf("Ep: %i, ID: %i, Cat: %i", art.epoch, ix, art.n_categories)))
        end
        sample = x[:, ix]
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
                # Current winner
                y_hat[ix] = art.labels[bmu]
                mismatch_flag = false
                break
            end
        end
        if mismatch_flag
            @debug "Mismatch"
            y_hat[ix] = -1
        end
    end

    return y_hat
end # classify(art::DDVFA, x::Array ; preprocessed=false)