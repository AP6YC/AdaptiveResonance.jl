"""
    common.jl

Description:
    Includes all of the unsupervised ARTMAP modules common code.
"""

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
        iter = get_iterator(art.opts, x)
        for i = iter
            # Update the iterator if necessary
            update_iter(art, iter, i)
            # Grab the sample slice
            # sample = get_sample(x, i)
            sample = x[:, i]
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
end # train!(art::ARTMAP, x::RealMatrix, y::IntegerVector, preprocessed::Bool=false)

# """
#     classify(art::ARTMAP, x::RealMatrix ; preprocessed::Bool=false, get_bmu::Bool=false)

# Predict categories of 'x' using the ARTMAP model.

# Returns predicted categories 'y_hat.'

# # Arguments
# - `art::ARTMAP`: supervised ARTMAP module to use for batch inference.
# - `x::RealMatrix`: the 2-D dataset containing columns of samples with rows of features.
# - `preprocessed::Bool=false`: flag, if the data has already been complement coded or not.
# - `get_bmu::Bool=false`, flag, if the model should return the best-matching-unit label in the case of total mismatch.

# # Examples
# ```julia-repl
# julia> my_DDVFA = DDVFA()
# DDVFA
#     opts: opts_DDVFA
#     ...
# julia> x, y = load_data()
# julia> train!(my_DDVFA, x)
# julia> y_hat = classify(my_DDVFA, y)
# ```
# """
# function classify(art::ARTMAP, x::RealMatrix ; preprocessed::Bool=false, get_bmu::Bool=false)
#     # Show a message if display is on
#     art.opts.display && @info "Testing $(typeof(art))"

#     # Preprocess the data
#     x = init_classify!(x, art, preprocessed)

#     # Data information and setup
#     n_samples = get_n_samples(x)

#     # Initialize the output vector
#     y_hat = zeros(Int, n_samples)

#     # Get the iterator based on the module options and data shape
#     iter = get_iterator(art.opts, x)
#     for ix = iter
#         # Update the iterator if necessary
#         update_iter(art, iter, ix)

#         # Grab the sample slice
#         sample = get_sample(x, ix)

#         # Get the classification
#         y_hat[ix] = classify(art, sample, preprocessed=true, get_bmu=get_bmu)
#     end

#     return y_hat
# end # classify(art::ARTMAP, x::RealMatrix ; preprocessed::Bool=false, get_bmu::Bool=false)
