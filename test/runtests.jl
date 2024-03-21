"""
    runtests.jl

# Description
The entry point to unit tests for the AdaptiveResonance.jl package.
"""

# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

using SafeTestsets

# -----------------------------------------------------------------------------
# SAFETESTSETS
# -----------------------------------------------------------------------------

@safetestset "All Test Sets" begin
    include("test_sets.jl")
end
