function random_test_train(dim::Integer, n_samples::Integer)
    art = DDVFA()
    features = rand(dim, n_samples)
    labels = rand(1:20, n_samples)

    train!(art, features, y=labels)
    # for ix = 1:n_samples
    #     sample = features[:, ix]
    #     label = labels[ix]
    #     train!(art, sample, y=label)
    # end
    return
end
