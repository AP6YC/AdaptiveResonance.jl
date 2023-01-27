"""
    common.jl

# Description:
Includes all of the unsupervised ART modules common code.
"""

# -----------------------------------------------------------------------------
# TYPES
# -----------------------------------------------------------------------------

"""
Abstract supertype of FuzzyART modules.
"""
abstract type AbstractFuzzyART <: ART end

# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------

"""
Train the ART model on a batch of data 'x' with optional supervisory labels 'y.'

# Arguments
- `art::ART`: the unsupervised ART model to train.
- `x::RealMatrix`: the 2-D dataset containing columns of samples with rows of features.
- `y::IntegerVector=Int[]`: optional, labels for simple supervisory training.
- `preprocessed::Bool=false`: optional, flag if the data has already been complement coded or not.
"""
function train!(art::ART, x::RealMatrix ; y::IntegerVector = Int[], preprocessed::Bool=false)
    # Show a message if display is on
    art.opts.display && @info "Training $(typeof(art))"

    # Flag for if training in supervised mode
    supervised = !isempty(y)

    # Data information and setup
    n_samples = get_n_samples(x)

    # Run the batch initialization procedure
    x = init_train!(x, art, preprocessed)

    # Initialize the output vector
    y_hat = zeros(Int, n_samples)

    # Learn until the stopping conditions
    art.epoch = 0
    while true
        # Increment the epoch and get the iterator
        art.epoch += 1
        iter = get_iterator(art.opts, n_samples)
        for i = iter
            # Update the iterator if necessary
            update_iter(art, iter, i)
            # Grab the sample slice
            sample = get_sample(x, i)
            # Select the label to pass to the incremental method
            local_y = supervised ? y[i] : 0
            # Train upon the sample and label
            y_hat[i] = train!(art, sample, y=local_y, preprocessed=true)
        end

        # Check stopping conditions
        if stopping_conditions(art)
            break
        end
    end
    return y_hat
end

# -----------------------------------------------------------------------------
# COMMON DOCUMENTATION
# -----------------------------------------------------------------------------

@doc raw"""
Train the ART model on a single sample of features 'x' with an optional supervisory label.

# Arguments
- `art::ART`: the unsupervised ART model to train.
- `x::RealVector`: the single sample feature vector to train upon.
- `y::Integer=0`: optional, a label for simple supervisory training.
- `preprocessed::Bool=false`: optional, flag if the data has already been complement coded or not.
"""
train!(art::ART, x::RealVector ; y::Integer=0, preprocessed::Bool=false)

@doc raw"""
Initializes the ART module for training with sample 'x' and optional label 'y', setting up the data configuration and instantiating the first category.

This function is used during the first training iteration when the ART module is empty.

# Arguments
- `art::ART`: the ART module to initialize.
- `x::RealVector`: the sample to use for initialization.
- `y::Integer=0`: the optional new label for the first weight of the ART module. If not specified, defaults the new label to 1.

# Examples
```julia-repl
julia> my_FuzzyART = FuzzyART()
FuzzyART
    opts: opts_FuzzyART
    ...
julia> initialize!(my_FuzzyART, [1, 2, 3, 4])
"""
initialize!(art::ART, x::RealVector ; y::Integer=0)

# COMMON DOC: FuzzyART initialization function
function initialize!(art::AbstractFuzzyART, x::RealVector ; y::Integer=0)
    # Set the threshold
    set_threshold!(art)
    # Initialize the feature dimension of the weights
    art.W = ARTMatrix{Float}(undef, art.config.dim_comp, 0)
    # Set the label to either the supervised label or 1 if unsupervised
    label = !iszero(y) ? y : 1
    # Create a category with the given label
    create_category!(art, x, label)
end