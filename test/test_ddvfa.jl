"""
    tt_ddvfa(opts::opts_DDVFA, train_x::Array)

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
    @info "Categories: $(art.n_categories)"
    @info "Weights: $total_cat"

    # # Calculate performance
    # perf = performance(y_hat, test_y)
    # println("Performance is ", perf)

    return art
end # tt_ddvfa(opts::opts_DDVFA, train_x::Array)

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
end # @testset "DDVFA"

@testset "GNFA" begin
    @info "GNFA Testing"
    Random.seed!(0)

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

    truth = Dict("single" => Dict("T" => 0.9988445088278305,
                                  "M" => 2.591300556893253),
                 "average" => Dict("T" => 0.41577750468594143,
                                   "M" => 1.322517210029363),
                 "complete" => Dict("T" => 0.04556971777638373,
                                    "M" => 0.13166315262229716),
                 "median" => Dict("T" => 0.3312241307874298,
                                  "M" => 1.3248965231497192),
                 "weighted" => Dict("T" => 0.533208585217186,
                                    "M" => 1.3855766656866793),
                 "centroid" => Dict("T" => 0.0,
                                    "M" => 0.0)
                )

    # Test every method and field name
    for method in methods
        results = Dict()
        for field_name in field_names
            results[field_name] = AdaptiveResonance.similarity(method, my_gnfa, field_name, local_sample, my_gnfa.opts.gamma_ref)
            @test isapprox(truth[method][field_name], results[field_name])
        end
        @info "Method: $method" results
    end

    # Check the error handling of the similarity function
    @test_throws ErrorException AdaptiveResonance.similarity("asdf", my_gnfa, "T", local_sample, my_gnfa.opts.gamma_ref)
    @test_throws ErrorException AdaptiveResonance.similarity("centroid", my_gnfa, "A", local_sample, my_gnfa.opts.gamma_ref)

end # @testset "GNFA"