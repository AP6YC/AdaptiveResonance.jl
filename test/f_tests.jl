using AdaptiveResonance
using Test
using CSV
using DrWatson


@testset "basics.jl" begin
    # Boilerplate tests to verify basic Julia use
    my_f(2, 1)
    @test 1 + 1 == 2
    @test foo(1, 1) == 2
    @test my_f(2, 1) == 5
    @test greet() == doGreet("World")
end


function ddvfa_example()

    CSV.read(datadir("art_data.csv"))

end

@testset "AdaptiveResonance.jl" begin


end

@testset "ARTMAP.jl" begin

    my_sfam = SFAM()

end