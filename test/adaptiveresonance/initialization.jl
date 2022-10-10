"""
    initialization.jl

# Description
Contains tests for module initialization.
"""

@testset "Initialization" begin
    # Default constructors
    fam = FAM()
    dam = DAM()
    sfam = SFAM()
    dvfa = DVFA()
    ddvfa = DDVFA()

    # Specify constructors
    fam_2 = FAM(opts_FAM())
    dam_2 = DAM(opts_DAM())
    sfam_2 = SFAM(opts_SFAM())
    dvfa_2 = DVFA(opts_DVFA())
    ddvfa_2 = DDVFA(opts_DDVFA())
end
