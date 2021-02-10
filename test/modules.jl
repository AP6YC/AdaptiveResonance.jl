@testset "Modules" begin
    # Default constructors
    fam = FAM()
    dam = DAM()
    sfam = SFAM()
    ddvfa = DDVFA()

    # Specify constructors
    fam_opts = opts_FAM()
    dam_opts = opts_DAM()
    sfam_opts = opts_SFAM()
    fam_2 = FAM(fam_opts)
    dam_2 = DAM(dam_opts)
    sfam_2 = SFAM(sfam_opts)
end