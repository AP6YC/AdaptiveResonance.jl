using LiveServer

# Make the documentation
include("make.jl")

# Serve the documentation for development
serve(dir="build")
