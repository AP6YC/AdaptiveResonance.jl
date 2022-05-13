"""
    serve.jl

Convenience script that serves the locally built documentation.
"""

using LiveServer

# Serve the documentation for development
serve(dir="build")
