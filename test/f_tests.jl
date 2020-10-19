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

end

@testset "AdaptiveResonance.jl" begin

    my_fam = FAM()
    my_dfam = DFAM()
    my_sfam = SFAM()
    my_ddvfa = DDVFA()

end

@testset "ARTMAP.jl" begin

    include("test_sfam.jl")
    sfam_example()

end