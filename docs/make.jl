using Documenter
using AdaptiveResonance

makedocs(
    modules=[AdaptiveResonance],
    format=Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        assets = [
            joinpath("assets", "favicon.ico")
        ]
    ),
    pages=[
        "Home" => "index.md",
        "Tutorial" => [
            "Guide" => "man/guide.md",
            "Examples" => "man/examples.md",
            "Contributing" => "man/contributing.md",
            "Index" => "man/full-index.md"
        ]
    ],
    repo="https://github.com/AP6YC/AdaptiveResonance.jl/blob/{commit}{path}#L{line}",
    sitename="AdaptiveResonance.jl",
    authors="Sasha Petrenko",
    # assets=String[],
)

deploydocs(
    repo="github.com/AP6YC/AdaptiveResonance.jl.git",
)
