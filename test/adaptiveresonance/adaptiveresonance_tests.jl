"""
    adaptiveresonance_tests.jl

# Description
Includes all of the AdaptiveResonance module tests.
"""

@testset "Exceptions" begin
    include("exceptions.jl")
end

@testset "Initialization" begin
    include("initialization.jl")
end

@testset "Common" begin
    include("common.jl")
end

@testset "Performance" begin
    include("performance.jl")
end
