"""
    tt_ddvfa(opts, train_x)

Trains and tests (tt) a DDVFA module on unlabled data train_x.
"""
function tt_ddvfa(opts::opts_DDVFA, train_x::Array)
    # Create the ART module, train, and classify
    art = DDVFA(opts)
    train!(art, train_x)
    y_hat = classify(art, train_x)
    # perf = performance(y_hat, data.test_y)

    # Total number of categories
    total_vec = [art.F2[i].n_categories for i = 1:art.n_categories]
    total_cat = sum(total_vec)
    println("Categories:", art.n_categories)
    println("Weights:", total_cat)

    return art

    # # Calculate performance
    # perf = performance(y_hat, test_y)
    # println("Performance is ", perf)
end

"""
    ddvfa_example()

Trains and tests multiple instances of DDVFA modules for full test coverage.
"""
# function ddvfa_example()
@testset "DDVFA" begin

    # Set the log level
    LogLevel(Logging.Info)

    # Parse the data
    data_file = "../data/art_data_rng.csv"
    train_x = readdlm(data_file, ',')
    train_x = permutedims(train_x)

    # Create the ART module, train, and classify
    @info "DDVFA Testing: Default Training"
    default_opts = opts_DDVFA()
    default_art = tt_ddvfa(default_opts, train_x)
    @info "DDVFA Testing: Default Complete"

    # Create the ART module, train, and classify with no display
    @info "DDVFA Testing: No Display Training"
    no_disp_opts = opts_DDVFA()
    no_disp_opts.display = false
    no_disp_art = tt_ddvfa(no_disp_opts, train_x)
    @info "DDVFA Testing: No Display Complete"

    @test default_art.W == no_disp_art.W
    # # View the profile as a flamegraph
    # ProfileVega.view()
end

@testset "GNFA" begin

    @info "GNFA Testing"

    # GNFA train and test
    my_gnfa = GNFA()
    # data = load_am_data(200, 50)
    data = load_iris("../data/Iris.csv")
    local_complement_code = AdaptiveResonance.complement_code(data.train_x)
    train!(my_gnfa, local_complement_code)

    # Similarity methods
    methods = ["single",
               "average",
               "complete",
               "median",
               "weighted",
               "centroid"]

    # Both field names
    field_names = ["T", "M"]

    # Compute a local sample for GNFA similarity method testing
    local_sample = local_complement_code[:, 1]

    # Compute the local activation and match
    AdaptiveResonance.activation_match!(my_gnfa, local_sample)

    # truth = Dict("single" => Dict("T" => 0.9999936533852463,
    #                               "M" => 483.5242679095584),
    #              "average" => Dict("T" => 0.745567955194466,
    #                                "M" => 428.88503692634504),
    #              "complete" => Dict("T" => 0.4422032564291216,
    #                                 "M" => 346.68735304043133),
    #              "median" => Dict("T" => 0.7510409115623551,
    #                               "M" => 431.3747398215685),
    #              "weighted" => Dict("T" => 0.8923661090763602,
    #                                 "M" => 437.9978445284187),
    #              "centroid" => Dict("T" => 1.5258610249962656e-5,
    #                                 "M" => 0.00390620422399044)
    #             )

    # Test every method and field name
    for method in methods
        println("Method: ", method)
        for field_name in field_names
            result = AdaptiveResonance.similarity(method, my_gnfa, field_name, local_sample, my_gnfa.opts.gamma_ref)
            println(field_name, ": ", result)
            # @test isapprox(truth[method][field_name], result)
        end
    end
end