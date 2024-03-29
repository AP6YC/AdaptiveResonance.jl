"""
    common.jl

# Description:
Includes all of the unsupervised ARTMAP modules common code.
"""

# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------

"""
    train!(art::ARTMAP, x::RealMatrix, y::IntegerVector, preprocessed::Bool=false)

Train the ARTMAP model on a batch of data 'x' with supervisory labels 'y.'

# Arguments
- `art::ARTMAP`: the supervised ARTMAP model to train.
- `x::RealMatrix`: the 2-D dataset containing columns of samples with rows of features.
- `y::IntegerVector`: labels for supervisory training.
- `preprocessed::Bool=false`: flag, if the data has already been complement coded or not.
"""
function train!(art::ARTMAP, x::RealMatrix, y::IntegerVector, preprocessed::Bool=false)
    # Show a message if display is on
    art.opts.display && @info "Training $(typeof(art))"

    # Data information and setup
    n_samples = length(y)

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
            label = y[i]
            # Train upon the sample and label
            y_hat[i] = train!(art, sample, label, preprocessed=true)
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
Train the supervised ARTMAP model on a single sample of features 'x' with supervisory label 'y'.

# Arguments
- `art::ARTMAP`: the supervised ART model to train.
- `x::RealVector`: the single sample feature vector to train upon.
- `y::Integer`: the label for supervisory training.
- `preprocessed::Bool=false`: optional, flag if the data has already been complement coded or not.
"""
train!(art::ARTMAP, x::RealVector, y::Integer ; preprocessed::Bool=false)

@doc raw"""
Initializes the supervised ARTMAP module for training with sample 'x' and label 'y', setting up the data configuration and instantiating the first category.

# Arguments
- `art::ARTMAP`: the ARTMAP module to initialize.
- `x::RealVector`: the sample to use for initialization.
- `y::Integer`: the initial supervised label.

# Examples
```julia-repl
julia> my_sfam = SFAM()
SFAM
    opts: opts_SFAM
    ...
julia> initialize!(my_SFAM, [1, 2, 3, 4])
"""
initialize!(art::ARTMAP, x::RealVector, y::Integer)
