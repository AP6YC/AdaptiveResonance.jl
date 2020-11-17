using AdaptiveResonance
using Test
using MLDatasets
using Logging
using DelimitedFiles

# Auxiliary generic functions for loading data, etc.
include("test_utils.jl")

@testset "DDVFA.jl" begin

    # DDVFA train and test functions
    include("test_ddvfa.jl")
    ddvfa_example()

    # GNFA train and test
    my_gnfa = GNFA()
    data = load_am_data(200, 50)
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

    # Test every method and field name
    for method in methods
        println("Method: ", method)
        for field_name in field_names
            result = AdaptiveResonance.similarity(method, my_gnfa, field_name, local_sample, my_gnfa.opts.gamma_ref)
            println(field_name, ": ", result)
        end
    end
end

@testset "AdaptiveResonance.jl" begin

    # Default constructors
    fam = FAM()
    dam = DAM()
    sfam = SFAM()
    ddvfa = DDVFA()

    # Specify constructors
    fam_opts = opts_FAM()
    dam_opts = opts_DAM()
    sfam_opts = opts_SFAM()
    fam_2 = FAM(fam_opts)
    dam_2 = DAM(dam_opts)
    sfam_2 = SFAM(sfam_opts)

end

@testset "ARTMAP.jl" begin

    # ARTMAP training and testing functions
    include("test_sfam.jl")
    data = load_am_data(200, 50)
    sfam_example(data)
    dam_example(data)

end
