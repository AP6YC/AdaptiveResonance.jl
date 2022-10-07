"""
    common.jl

Description:
    Includes all of the unsupervised ART modules common code.
"""

# --------------------------------------------------------------------------- #
# FUNCTIONS
# --------------------------------------------------------------------------- #

"""
Train the ART model on a batch of data 'x' with optional supervisory labels 'y.'

# Arguments
- `art::ART`: the unsupervised ART model to train.
- `x::RealMatrix`: the 2-D dataset containing columns of samples with rows of features.
- `y::IntegerVector=Vector{Int}()`: optional, labels for simple supervisory training.
- `preprocessed::Bool=false`: optional, flag if the data has already been complement coded or not.
"""
function train!(art::ART, x::RealMatrix ; y::IntegerVector = Vector{Int}(), preprocessed::Bool=false)
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
        iter = get_iterator(art.opts, x)
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
end # train!(art::ART, x::RealMatrix ; y::IntegerVector = Vector{Int}(), preprocessed::Bool=false)

# --------------------------------------------------------------------------- #
# COMMON DOCUMENTATION
# --------------------------------------------------------------------------- #

@doc raw"""
Train the ART model on a single sample of features 'x' with an optional supervisory label.

# Arguments
- `art::ART`: the unsupervised ART model to train.
- `x::RealVector`: the single sample feature vector to train upon.
- `y::Integer=0`: optional, a label for simple supervisory training.
- `preprocessed::Bool=false`: optional, flag if the data has already been complement coded or not.
"""
train!(art::ART, x::RealVector ; y::Integer=0, preprocessed::Bool=false)
