"""
    runtests.jl

# Description
The entry point to unit tests for the AdaptiveResonance.jl package.
"""

# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

# using SafeTestsets

# -----------------------------------------------------------------------------
# SAFETESTSETS
# -----------------------------------------------------------------------------

# @safetestset "All Test Sets" begin
using Test

@testset verbose=true "All tests" begin
    include("test_sets.jl")
end
