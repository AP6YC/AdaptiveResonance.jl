"""
    ddvfa.jl

# Description
DDVFA-specific test sets.
"""

@testset "Convenience functions" begin
    my_art = DDVFA()
    train!(my_art, data.train_x)

    # Convenience functions
    W = AdaptiveResonance.get_W(my_art)
    n_vec = AdaptiveResonance.get_n_weights_vec(my_art)
    n_weights = AdaptiveResonance.get_n_weights(my_art)

    n_F2 = length(my_art.F2)

    # Test these values
    @test ndims(W) == 1         # W is a list
    @test length(W) == n_F2     # W has n_F2 weights
    @test n_vec isa Vector      # n_vec is a vector
    @test length(n_vec) == n_F2 # n_vec describes n_F2 nodes
    @test n_weights isa Int     # n_weights is one number
end
