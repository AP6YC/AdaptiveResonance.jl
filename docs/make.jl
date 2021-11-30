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
        "Getting Started" => [
            "getting-started/whatisart.md",
            "getting-started/basic-example.md",
        ],
        "Tutorial" => [
            "Guide" => "man/guide.md",
            "Examples" => "man/examples.md",
            "Modules" => "man/modules.md",
            "Contributing" => "man/contributing.md",
            "Index" => "man/full-index.md",
        ],
    ],
    repo="https://github.com/AP6YC/AdaptiveResonance.jl/blob/{commit}{path}#L{line}",
    sitename="AdaptiveResonance.jl",
    authors="Sasha Petrenko",
    # assets=String[],
)

deploydocs(
    repo="github.com/AP6YC/AdaptiveResonance.jl.git",
    devbranch="develop",
)
