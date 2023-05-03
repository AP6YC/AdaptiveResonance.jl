"""
    make.jl

# Description
This file builds the documentation for the AdaptiveResonance.jl package
using Documenter.jl and other tools.
"""

# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

using
    Documenter,
    DemoCards,
    Logging,
    Pkg

# -----------------------------------------------------------------------------
# SETUP
# -----------------------------------------------------------------------------

# Common variables of the script
PROJECT_NAME = "AdaptiveResonance"
DOCS_NAME = "docs"

# Fix GR headless errors
ENV["GKSwstype"] = "100"

# Get the current workind directory's base name
current_dir = basename(pwd())
@info "Current directory is $(current_dir)"

# If using the CI method `julia --project=docs/ docs/make.jl`
#   or `julia --startup-file=no --project=docs/ docs/make.jl`
if occursin(PROJECT_NAME, current_dir)
    push!(LOAD_PATH, "../src/")
# Otherwise, we are already in the docs project and need to dev the above package
elseif occursin(DOCS_NAME, current_dir)
    Pkg.develop(path="..")
# Otherwise, building docs from the wrong path
else
    error("Unrecognized docs setup path")
end

# Inlude the local package
using AdaptiveResonance

# using JSON
if haskey(ENV, "DOCSARGS")
    for arg in split(ENV["DOCSARGS"])
        (arg in ARGS) || push!(ARGS, arg)
    end
end

# -----------------------------------------------------------------------------
# DOWNLOAD LARGE ASSETS
# -----------------------------------------------------------------------------

# Point to the raw FileStorage location on GitHub
top_url = raw"https://media.githubusercontent.com/media/AP6YC/FileStorage/main/AdaptiveResonance/"

# List all of the files that we need to use in the docs
files = [
    "header.png",
    "art.png",
    "artmap.png",
    "ddvfa.png",
]

# Make a destination for the files, accounting for when folder is AdaptiveResonance.jl
if basename(pwd()) == PROJECT_NAME || basename(pwd()) == PROJECT_NAME * ".jl"
    download_folder = joinpath(DOCS_NAME, "src", "assets", "downloads")
else
    download_folder = joinpath("src", "assets", "downloads")
end
mkpath(download_folder)
download_list = []

# Download the files one at a time
for file in files
    # Point to the correct file that we wish to download
    src_file = top_url * file * "?raw=true"
    # Point to the correct local destination file to download to
    dest_file = joinpath(download_folder, file)
    # Add the file to the list that we will append to assets
    push!(download_list, dest_file)
    # If the file isn't already here, download it
    if !isfile(dest_file)
        download(src_file, dest_file)
        @info "Downloaded $dest_file, isfile: $(isfile(dest_file))"
    else
        @info "File already exists: $dest_file"
    end
end

# Downloads debugging
detailed_logger = Logging.ConsoleLogger(stdout, Info, show_limited=false)
with_logger(detailed_logger) do
    @info "Current working directory is $(pwd())"
    @info "Assets folder is:" readdir(joinpath(pwd(), "src", "assets"), join=true)
    full_download_folder = joinpath(pwd(), "src", "assets", "downloads")
    @info "Downloads folder exists: $(isdir(full_download_folder))"
    if isdir(download_folder)
        @info "Downloads folder contains:" readdir(full_download_folder, join=true)
    end
end

# -----------------------------------------------------------------------------
# GENERATE
# -----------------------------------------------------------------------------

# Generate the demo files
# this is the relative path to docs/
demopage, postprocess_cb, demo_assets = makedemos("examples")

assets = [
    joinpath("assets", "favicon.ico"),
]

# @info "Favicon?"
# @info isfile(joinpath("assets", "favicon.ico"))

# # Add the downloaded files to the assets list
# for file in files
#     local_file = joinpath("assets", "downloads", file)
#     @info isfile(local_file)
#     push!(assets, local_file)
# end

# if there are generated css assets, pass it to Documenter.HTML
isnothing(demo_assets) || (push!(assets, demo_assets))

# Make the documentation
makedocs(
    modules=[AdaptiveResonance],
    format=Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        assets = assets,
    ),
    pages=[
        "Home" => "index.md",
        "Getting Started" => [
            "getting-started/whatisart.md",
            "getting-started/basic-example.md",
        ],
        "Tutorial" => [
            "Guide" => "man/guide.md",
            demopage,
            # "Examples" => "man/examples.md",
            "Modules" => "man/modules.md",
            "Contributing" => "man/contributing.md",
            "Index" => "man/full-index.md",
            "Internals" => "man/dev-index.md",
        ],
    ],
    repo="https://github.com/AP6YC/AdaptiveResonance.jl/blob/{commit}{path}#L{line}",
    sitename="AdaptiveResonance.jl",
    authors="Sasha Petrenko",
    # assets=String[],
)

# 3. postprocess after makedocs
postprocess_cb()

# a workdaround to github action that only push preview when PR has "push_preview" labels
# issue: https://github.com/JuliaDocs/Documenter.jl/issues/1225
# function should_push_preview(event_path = get(ENV, "GITHUB_EVENT_PATH", nothing))
#     event_path === nothing && return false
#     event = JSON.parsefile(event_path)
#     haskey(event, "pull_request") || return false
#     labels = [x["name"] for x in event["pull_request"]["labels"]]
#     return "push_preview" in labels
#  end

# -----------------------------------------------------------------------------
# DEPLOY
# -----------------------------------------------------------------------------

deploydocs(
    repo="github.com/AP6YC/AdaptiveResonance.jl.git",
    devbranch="develop",
    # push_preview = should_push_preview(),
)
