"""
    tt_ddvfa(opts, train_x)

Trains and tests (tt) a DDVFA module on unlabled data train_x.
"""
function tt_ddvfa(opts::opts_DDVFA, train_x::Array)
    # Create the ART module, train, and classify
    art = DDVFA(opts)
    train!(art, train_x)

    # Total number of categories
    total_vec = [art.F2[i].n_categories for i = 1:art.n_categories]
    total_cat = sum(total_vec)
    println("Categories:", art.n_categories)
    println("Weights:", total_cat)

    # # Calculate performance
    # perf = performance(y_hat, test_y)
    # println("Performance is ", perf)
end


"""
    ddvfa_example()

Trains and tests multiple instances of DDVFA modules for full test coverage.
"""
function ddvfa_example()

    # Set the log level
    LogLevel(Logging.Info)

    # Parse the data
    data_file = "../data/art_data_rng.csv"
    train_x = readdlm(data_file, ',')
    train_x = permutedims(train_x)

    # Create the ART module, train, and classify
    @info "DDVFA Testing: Default Training"
    default_opts = opts_DDVFA()
    tt_ddvfa(default_opts, train_x)
    @info "DDVFA Testing: Default Complete"

    # Create the ART module, train, and classify with no display
    @info "DDVFA Testing: No Display Training"
    no_disp_opts = opts_DDVFA()
    no_disp_opts.display = false
    tt_ddvfa(no_disp_opts, train_x)
    @info "DDVFA Testing: No Display Complete"

    # # View the profile as a flamegraph
    # ProfileVega.view()
end
