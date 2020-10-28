using AdaptiveResonance
using Test
using MLDatasets
using Logging
using DelimitedFiles

@testset "basics.jl" begin
    # Boilerplate tests to verify basic Julia use
    @test my_f(2, 2) == 6
    @test 1 + 1 == 2
    @test foo(1, 1) == 2
    @test my_f(2, 1) == 5
    @test greet() == doGreet("World")
end


@testset "DDVFA.jl" begin

    include("test_ddvfa.jl")
    ddvfa_example()
    my_gnfa = GNFA()

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

    include("test_sfam.jl")
    data = load_am_data()
    sfam_example(data)
    dam_example(data)

end