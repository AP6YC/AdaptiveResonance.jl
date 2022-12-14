"""
    performance.jl

# Description
A test of the performance of every ART and ARTMAP module.
"""

@testset "Training Test" begin

    @info "------- Training test -------"

    # All ART modules
    arts = ADAPTIVERESONANCE_MODULES
    n_arts = length(arts)

    # All common ART options
    art_opts = [
        (display = true,),
        # (display = false,),
    ]

    # Specific ART options
    art_specifics = Dict(
        DDVFA => [
            (gamma_normalization=true,),
            (gamma_normalization=false,),
        ],
        FuzzyART => [
            (gamma_normalization=true,),
            (gamma_normalization=false,),
        ],
    )

    # All test option permutations
    test_opts = [
        (get_bmu = true,),
        (get_bmu = false,)
    ]
    n_test_opts = length(test_opts)

    # Performance baseline for all algorithms
    perf_baseline = 0.7

    # Iterate over all ART modules
    for ix = 1:n_arts
        # Iterate over all test options
        for jx = 1:n_test_opts
            # If we are testing a module with different options, merge
            if haskey(art_specifics, arts[ix])
                local_art_opts = vcat(art_opts, art_specifics[arts[ix]])
            else
                local_art_opts = art_opts
            end
            # Iterate over all options
            for kx in eachindex(local_art_opts)
                # Only do the unsupervised method if we have an ART module (not ARTMAP)
                if arts[ix] isa ART
                    # Unsupervised
                    train_test_art(arts[ix](;local_art_opts[kx]...), data; test_opts=test_opts[jx])
                end
                # Supervised
                @test train_test_art(arts[ix](;local_art_opts[kx]...), data; supervised=true, test_opts=test_opts[jx]) >= perf_baseline
            end
        end
    end
end
