using Documenter, AdaptiveResonance

makedocs(;
    modules=[AdaptiveResonance],
    format=Documenter.HTML(),
    # pages=[
    #     "Home" => "index.md",
    # ],
    repo="https://github.com/AP6YC/AdaptiveResonance.jl/blob/{commit}{path}#L{line}",
    sitename="AdaptiveResonance.jl",
    authors="Sasha Petrenko",
    assets=String[],
)

deploydocs(;
    repo="github.com/AP6YC/AdaptiveResonance.jl",
)
