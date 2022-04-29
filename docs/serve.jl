"""
    serve.jl

This is a convenience script for docs  development that
"""

using LiveServer

# Make the documentation
include("make.jl")

# Serve the documentation for development
serve(dir="build")
