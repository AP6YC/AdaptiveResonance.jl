using AdaptiveResonance
using Test
using Logging
using DelimitedFiles

# Set the log level
LogLevel(Logging.Info)

# Auxiliary generic functions for loading data, etc.
include("test_utils.jl")

# Load the data and test across all supervised modules
data = load_iris("../data/Iris.csv")

@testset "common.jl" begin
    @info "------- Common Code Tests -------"
    # Example arrays
    three_by_two = [1 2; 3 4; 5 6]

    # Test DataConfig constructors
    dc1 = DataConfig()                  # Default constructor
    dc2 = DataConfig(0, 1, 2)           # When min and max are same across all features
    dc3 = DataConfig([0, 1], [2, 3])    # When min and max differ across features
    dc4 = DataConfig(three_by_two)      # When a data matrix is provided

    # Test get_n_samples
    @test get_n_samples([1,2,3]) == 1           # 1-D array case
    @test get_n_samples(three_by_two) == 2      # 2-D array case

    # Test breaking situations
    @test_throws ErrorException performance([1,2],[1,2,3])
    @test_logs (:warn,) AdaptiveResonance.data_setup!(dc3, three_by_two)
    bad_config =  DataConfig(1, 0, 3)
    @test_throws ErrorException linear_normalization(three_by_two, config=bad_config)
end # @testset "common.jl"

@testset "constants.jl" begin
    @info "------- Constants Tests -------"
    ddvfa_methods = [
        "single",
        "average",
        "complete",
        "median",
        "weighted",
        "centroid"
    ]
    @test AdaptiveResonance.DDVFA_METHODS == ddvfa_methods
end # @testset "constants.jl"

@testset "AdaptiveResonance.jl" begin
    # Module loading
    include("modules.jl")
end # @testset "AdaptiveResonance.jl"

@testset "Train Test" begin
    # All ART modules
    arts = [
        FuzzyART,
        DVFA,
        DDVFA,
        SFAM,
        DAM,
    ]
    n_arts = length(arts)

    # All common ART options
    art_opts = [
        (display = true,),
        # (display = false,),
    ]

    # Specific ART options
    art_specifics = Dict(
        DDVFA => [
            (gamma_normalization=true,),
            (gamma_normalization=false,),
        ],
        FuzzyART => [
            (gamma_normalization=true,),
            (gamma_normalization=false,),
        ],
    )

    # All test option permutations
    test_opts = [
        (get_bmu = true,),
        (get_bmu = false,)
    ]
    n_test_opts = length(test_opts)

    @info "-------------- BEGIN TRAIN TEST --------------"
    # Performance baseline for all algorithms
    perf_baseline = 0.7

    # Iterate over all ART modules
    for ix = 1:n_arts
        # Iterate over all test options
        for jx = 1:n_test_opts
            # If we are testing a module with different options, merge
            if haskey(art_specifics, arts[ix])
                local_art_opts = vcat(art_opts, art_specifics[arts[ix]])
            else
                local_art_opts = art_opts
            end
            # Iterate over all options
            for kx = 1:length(local_art_opts)
                # Only do the unsupervised method if we have an ART module (not ARTMAP)
                if arts[ix] isa ART
                    # Unsupervised
                    train_test_art(arts[ix](;local_art_opts[kx]...), data; test_opts=test_opts[jx])
                end
                # Supervised
                @test train_test_art(arts[ix](;local_art_opts[kx]...), data; supervised=true, test_opts=test_opts[jx]) >= perf_baseline
            end
        end
    end

    @info "-------------- END TRAIN TEST --------------"
end # @testset "Train Test"

@testset "kwargs" begin
    @info "--------- KWARGS TEST ---------"

    arts = [
        FuzzyART,
        DVFA,
        DDVFA,
        SFAM,
        DAM
    ]

    for art in arts
        art_module = art(alpha=1e-3, display=false)
    end

    @info "--------- END KWARGS TEST ---------"
end # @testset "kwargs"

@testset "FuzzyART" begin
    @info "------- FuzzyART Testing -------"

    # FuzzyART train and test
    my_FuzzyART = FuzzyART()
    # local_complement_code = AdaptiveResonance.complement_code(data.train_x)
    # train!(my_FuzzyART, local_complement_code, preprocessed=true)
    train!(my_FuzzyART, data.train_x)

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

    # Compute a local sample for FuzzyART similarity method testing
    # local_sample = local_complement_code[:, 1]
    # local_complement_code = AdaptiveResonance.complement_code(data.train_x)
    # local_sample = data.train_x[:, 1]
    local_sample = AdaptiveResonance.complement_code(data.train_x[:, 1], config=my_FuzzyART.config)

    # Compute the local activation and match
    # AdaptiveResonance.activation_match!(my_FuzzyART, local_sample)

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
    #         results[field_name] = AdaptiveResonance.similarity(method, my_FuzzyART, field_name, local_sample, my_FuzzyART.opts.gamma_ref)
    #         @test isapprox(truth[method][field_name], results[field_name])
    #     end
    #     @info "Method: $method" results
    # end

    # Check the error handling of the similarity function
    # Access the wrong similarity metric keyword ("asdf")
    @test_throws ErrorException AdaptiveResonance.similarity("asdf", my_FuzzyART, "T", local_sample, my_FuzzyART.opts.gamma_ref)
    # Access the wrong output function ("A")
    @test_throws ErrorException AdaptiveResonance.similarity("centroid", my_FuzzyART, "A", local_sample, my_FuzzyART.opts.gamma_ref)

end # @testset "FuzzyART"

@testset "ARTSCENE.jl" begin
    # ARTSCENE training and testing
    include("test_artscene.jl")
end # @testset "ARTSCENE.jl"
