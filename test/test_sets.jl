"""
    test_sets.jl

# Description
The main collection of tests for the AdaptiveResonance.jl package.
This file loads common utilities and aggregates all other unit tests files.
"""

using AdaptiveResonance
using Test
using Logging
using DelimitedFiles

# Set the log level
LogLevel(Logging.Info)

# Auxiliary generic functions for loading data, etc.
include("test_utils.jl")

# Load the data and test across all supervised modules
data = load_iris("data/Iris.csv")


# @testset "AdaptiveResonance.jl" begin
#     # Module loading
#     include("modules.jl")
# end # @testset "AdaptiveResonance.jl"

@testset "AdaptiveResonance" begin
    @info "------- ADAPTIVERESONANCE TESTS -------"
    include("adaptiveresonance/adaptiveresonance_tests.jl")
end

@testset "ART" begin
    @info "------- ART TESTS -------"
    include("art/art_tests.jl")
end

@testset "ARTMAP" begin
    @info "------- ARTMAP TESTS -------"
    include("artmap/artmap_tests.jl")
end
