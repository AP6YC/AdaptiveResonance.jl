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
The ART modules are driven by options, which are themselves mutable keyword argument structs from the [Parameters.jl](https://github.com/mauro3/Parameters.jl) package.

## ART Modules

The AdaptiveResonance package is designed for maximum flexibility for scientific research, even though this may come at the cost of learning instability if misused.
Because of the diversity of ART modules, the package is structured around instantiating separate modules and using them for training and inference.
Due to this diversity, each module has its own options struct with keyword arguments.
These options have default values driven by standards in their respective literatures, so the ART modules may be used immediately without any customization.
Furthermore, these options are mutable, so they may be modified before module instantiation, before training, or even after training.

For example, you can get going with the default options by creating an ART module with the default constructor:

```julia
my_art = DDVFA()
```

If you want to change the parameters before construction, you can create an options struct, modify it, then instantiate your ART module with it:

```julia
my_art_opts = opts_DDVFA()
my_art_opts.gamma = 3
my_art = DDVFA(my_art_opts)
```

The options are objects from the [Parameters.jl](https://github.com/mauro3/Parameters.jl) project,

You can even modify the parameters on the fly after the ART module has been instantiated by directly modifying the options within the module:

```julia
my_art = DDVFA()
my_art.opts.gamma = 3
```
