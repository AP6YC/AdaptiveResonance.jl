# Package Guide

## Installation

The AdaptiveResonance package can be installed using the Julia package manager.
From the Julia REPL, type ']' to enter the Pkg REPL mode and run

```julia
pkg> add AdaptiveResonance
```

Alternatively, it can be added to/ your environment in a script with

```julia
using Pkg
Pkg.add("AdaptiveResonance")
```

## Usage

The AdaptiveResonance package is built upon ART modules that contain all of the state information during training and inference.
The ART modules are driven by options, which are themselves mutable keyword argument structs from the [Parameters](https://github.com/mauro3/Parameters.jl) package.