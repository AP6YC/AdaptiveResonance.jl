"""
    combo.jl

This is a convenience script for docs development that makes and live serves the docs locally.
"""


# Make the documentation
include("make.jl")

# Host the documentation locally
include("serve.jl")
