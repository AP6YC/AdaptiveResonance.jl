using AdaptiveResonance
using Test
using CSV
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
    dfam = DFAM()
    sfam = SFAM()
    ddvfa = DDVFA()

    # Specify constructors
    fam_opts = opts_FAM()
    dfam_opts = opts_DFAM()
    sfam_opts = opts_SFAM()
    fam_2 = FAM(fam_opts)
    dfam_2 = DFAM(dfam_opts)
    sfam_2 = SFAM(sfam_opts)
end

@testset "ARTMAP.jl" begin

    include("test_sfam.jl")
    sfam_example()

end