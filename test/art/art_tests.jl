"""
    art_tests.jl

# Description
Includes all of the ART module tests.
"""

@testset "DDVFA" begin
    include("ddvfa.jl")
end

@testset "FuzzyART" begin
    include("fuzzyart.jl")
end
