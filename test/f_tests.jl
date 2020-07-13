using AdaptiveResonance
using Test

function foo(a, b)
    return a + b
end

my_f(2, 1)

@testset "AdaptiveResonance.jl" begin
    # Write your own tests here.
    @test 1 + 1 == 2
    @test foo(1, 1) == 2
    @test my_f(2, 1) == 5
    @test greet() == doGreet("World")
end