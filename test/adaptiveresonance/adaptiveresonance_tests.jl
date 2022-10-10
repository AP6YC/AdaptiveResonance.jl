"""
    adaptiveresonance_tests.jl

# Description
Includes all of the AdaptiveResonance module tests.
"""

@testset "Initialization" begin
    include("initialization.jl")
end

@testset "Common" begin
    include("common.jl")
end

@testset "Performance" begin
    include("performance.jl")
end
