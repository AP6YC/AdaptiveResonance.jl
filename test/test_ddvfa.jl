"""
    tt_ddvfa(opts::opts_DDVFA, train_x::Array)

Trains and tests (tt) a DDVFA module on unlabeled data train_x.
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

    return art
end # tt_ddvfa(opts::opts_DDVFA, train_x::Array)

@testset "DDVFA Sequential" begin
    @info "------- DDVFA Sequential -------"

    # Initialize the ART module
    art = DDVFA()
    # Turn off display for sequential training/testing
    art.opts.display = false
    # Set up the data manually because the module can't infer from single samples
    data_setup!(art.config, data.train_x)

    # Get the dimension and size of the data
    dim, n_samples = get_data_shape(data.train_x)
    y_hat_train = zeros(Int64, n_samples)
    dim_test, n_samples_test = get_data_shape(data.test_x)
    y_hat = zeros(Int64, n_samples_test)
    y_hat_bmu = zeros(Int64, n_samples_test)

    # Iterate over all examples sequentially
    for i = 1:n_samples
        y_hat_train[i] = train!(art, data.train_x[:, i], y=[data.train_y[i]])
    end

    # Iterate over all test samples sequentially
    for i = 1:n_samples_test
        y_hat[i] = classify(art, data.test_x[:, i])
        y_hat_bmu[i] = classify(art, data.test_x[:, i], get_bmu=true)
    end

    # Calculate performance
    perf_train = performance(y_hat_train, data.train_y)
    perf_test = performance(y_hat, data.test_y)
    perf_test_bmu = performance(y_hat_bmu, data.test_y)

    # Test the permance above a baseline number
    perf_baseline = 0.8
    @test perf_train >= perf_baseline
    @test perf_test >= perf_baseline
    @test perf_test_bmu >= perf_baseline

    @info "DDVFA Training Perf: $perf_train"
    @info "DDVFA Testing Perf: $perf_test"
    @info "DDVFA Testing BMU Perf: $perf_test_bmu"
end

@testset "DDVFA Supervised" begin
    @info "------- DDVFA Supervised -------"

    # Train and classify
    art = DDVFA()
    y_hat_train = train!(art, data.train_x, y=data.train_y)
    y_hat = classify(art, data.test_x)
    y_hat_bmu = classify(art, data.test_x, get_bmu=true)

    # Calculate performance
    perf_train = performance(y_hat_train, data.train_y)
    perf_test = performance(y_hat, data.test_y)
    perf_test_bmu = performance(y_hat_bmu, data.test_y)

    # Test the performances with a baseline number
    perf_baseline = 0.8
    @test perf_train >= perf_baseline
    @test perf_test >= perf_baseline
    @test perf_test_bmu >= perf_baseline

    # Log the results
    @info "DDVFA Training Perf: $perf_train"
    @info "DDVFA Testing Perf: $perf_test"
    @info "DDVFA Testing BMU Perf: $perf_test_bmu"
end # @testset "DDVFA Supervised"

@testset "DDVFA" begin
    # Parse the data
    data_file = "../data/art_data_rng.csv"
    train_x = readdlm(data_file, ',')
    train_x = permutedims(train_x)

    # Create the ART module, train, and classify
    @info " ------- DDVFA Testing: Default Training -------"
    default_opts = opts_DDVFA()
    default_ddvfa = tt_ddvfa(default_opts, train_x)
    @info "DDVFA Testing: Default Complete"

    # Create the ART module, train, and classify with no display
    @info "------- DDVFA Testing: No Display Training -------"
    no_disp_opts = opts_DDVFA()
    no_disp_opts.display = false
    no_disp_ddvfa = tt_ddvfa(no_disp_opts, train_x)
    @info "DDVFA Testing: No Display Complete"

    # Test that the resulting weights are equivalent
    @test get_W(default_ddvfa) == get_W(no_disp_ddvfa)
end # @testset "DDVFA"

@testset "GNFA" begin
    @info "------- GNFA Testing -------"

    # GNFA train and test
    my_gnfa = GNFA()
    # local_complement_code = AdaptiveResonance.complement_code(data.train_x)
    # train!(my_gnfa, local_complement_code, preprocessed=true)
    train!(my_gnfa, data.train_x)

    # Similarity methods
    methods = [
        "single",
        "average",
        "complete",
        "median",
        "weighted",
        "centroid"
    ]

    # Both field names
    field_names = ["T", "M"]

    # Compute a local sample for GNFA similarity method testing
    # local_sample = local_complement_code[:, 1]
    # local_complement_code = AdaptiveResonance.complement_code(data.train_x)
    # local_sample = data.train_x[:, 1]
    local_sample = AdaptiveResonance.complement_code(data.train_x[:, 1], config=my_gnfa.config)

    # Compute the local activation and match
    # AdaptiveResonance.activation_match!(my_gnfa, local_sample)

    # # Declare the true activation and match magnitudes
    # truth = Dict(
    #     "single" => Dict(
    #         "T" => 0.9988714513100155,
    #         "M" => 2.6532834139109758
    #     ),
    #     "average" => Dict(
    #         "T" => 0.33761483787933894,
    #         "M" => 1.1148764060015297
    #     ),
    #     "complete" => Dict(
    #         "T" => 0.018234409874338647,
    #         "M" => 0.07293763949735459
    #     ),
    #     "median" => Dict(
    #         "T" => 0.2089217851518073,
    #         "M" => 0.835687140607229
    #     ),
    #     "weighted" => Dict(
    #         "T" => 0.5374562506748786,
    #         "M" => 1.4396083090159748
    #     ),
    #     "centroid" => Dict(
    #         "T" => 0.0,
    #         "M" => 0.0
    #     )
    # )

    # # Test every method and field name
    # for method in methods
    #     results = Dict()
    #     for field_name in field_names
    #         results[field_name] = AdaptiveResonance.similarity(method, my_gnfa, field_name, local_sample, my_gnfa.opts.gamma_ref)
    #         @test isapprox(truth[method][field_name], results[field_name])
    #     end
    #     @info "Method: $method" results
    # end

    # Check the error handling of the similarity function
    # Access the wrong similarity metric keyword ("asdf")
    @test_throws ErrorException AdaptiveResonance.similarity("asdf", my_gnfa, "T", local_sample, my_gnfa.opts.gamma_ref)
    # Access the wrong output function ("A")
    @test_throws ErrorException AdaptiveResonance.similarity("centroid", my_gnfa, "A", local_sample, my_gnfa.opts.gamma_ref)

end # @testset "GNFA"
