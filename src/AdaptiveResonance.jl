"""
Main module for `AdaptiveResonance.jl`, a Julia package of adaptive resonance theory algorithms.

This module exports all of the ART modules, options, and utilities used by the `AdaptiveResonance.jl package.`
For full usage, see the official guide at https://ap6yc.github.io/AdaptiveResonance.jl/dev/man/guide/.

# Basic Usage

Install and import the package in a script with

```julia
using Pkg
Pkg.add("AdaptiveResonance")
using AdaptiveResonance
```

then create an ART module with default options

```julia
my_art = DDVFA()
```

or custom options via keyword arguments

```julia
my_art = DDVFA(rho_ub=0.45, rho_ub=0.7)
```

Train all models with `train!` and conduct inference with `classify`.
In batch, samples are interpreted in the Julia column-major fashion with dimensions `(n_dim, n_samples)` (i.e., columns are samples).

Train unsupervised ART modules incrementally or in batch with optional labels as a keyword argument `y`

```julia
# Load your data somehow
samples, labels = load_some_data()

# Unsupervised batch
train!(my_art, samples)

# Supervised batch
train!(my_art, samples, y=labels)

# Unsupervised incremental
for ix in eachindex(labels)
    train!(my_art, samples[:, ix])
end

# Supervised incremental
for ix in eachindex(labels)
    train!(my_art, samples[:, ix], y=labels[ix])
end
```

Train supervised ARTMAP with positional arguments

```julia
my_artmap = SFAM()
train!(my_artmap, samples, labels)
```

With either module, conduct inference with `classify(art, samples)`

```julia
# Batch inference
y_hat = classify(my_art, test_samples)

# Incremental inference
for ix in eachindex(test_labels)
    y_hat[ix] = classify(my_artmap, test_samples[:, ix])
end
```

# Imports

The following names are imported by the package as dependencies:
$(IMPORTS)

# Exports

The following names are exported and available when `using` the package:
$(EXPORTS)
"""
module AdaptiveResonance

# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

# Usings/imports for the whole package declared once

# Full usings (which supports comma-separated import notation)
using
    DocStringExtensions,    # Docstring utilities
    ElasticArrays,          # Fast resizable arrays
    Logging,                # Logging utils used as main method of terminal reporting
    NumericalTypeAliases,   # Abstract type aliases
    Parameters,             # ARTopts are parameters (@with_kw)
    ProgressBars            # Provides progress bars for training and inference

# Specific identifiers
using LinearAlgebra: norm   # Trace and norms

# Medians and mean for linkage methods
using Statistics: median as statistics_median
using Statistics: mean as statistics_mean

# Precompile concrete type methods
using PrecompileSignatures: @precompile_signatures

# -----------------------------------------------------------------------------
# INCLUDES
# -----------------------------------------------------------------------------

# Include all files
# include("common.jl")        # Objects shared by all modules
# Common objects and functions shared by all modules
include("lib/lib.jl")

# Exported constant for the version of the package
include("version.jl")

# Supervised ART modules
include("ARTMAP/ARTMAP.jl")

# Unsupervised ART modules
include("ART/ART.jl")

# -----------------------------------------------------------------------------
# DERIVATIVE TYPES AND CONSTANTS
# -----------------------------------------------------------------------------

"""
A combined list of all unsupervised ART and supervised ARTMAP modules from the `AdaptiveResonance.jl` package.
"""
const ADAPTIVERESONANCE_MODULES = [
    ART_MODULES;
    ARTMAP_MODULES;
]

# -----------------------------------------------------------------------------
# EXPORTS
# -----------------------------------------------------------------------------

# Export all public names
export

    # Abstract types
    ARTOpts,        # All module options are ARTOpts
    ARTModule,      # All modules are ART modules
    ART,            # ART modules (unsupervised)
    ARTMAP,         # ARTMAP modules (supervised)

    # Algorithmic functions
    train!,         # Train models: train!(art, data, y=[])
    classify,       # Inference: classify(art, data)
    performance,    # Classification accuracy: performance(y, y_hat)

    # Common structures
    DataConfig,     # ART data configs (feature ranges, dimension, etc.)
    data_setup!,    # Correctly set up an ART data configuration

    # Common utility functions
    complement_code,            # Map x -> [x, 1 - x] and normalize to [0, 1]
    get_data_characteristics,   # Get characteristics of x, used by data configs
    linear_normalization,       # Normalize x to [0, 1]

    # ART (unsupervised)
    FuzzyART, opts_FuzzyART,
    DDVFA, opts_DDVFA, get_W,
    DVFA, opts_DVFA,
    # ART variants
    GammaNormalizedFuzzyART, opts_GammaNormalizedFuzzyART,

    # ARTMAP (supervised)
    FAM, opts_FAM,
    SFAM, opts_SFAM,
    # ARTMAP variants
    DAM, opts_DAM,

    # Useful constants
    ART_MODULES,                # List of (default) unsupervised ART modules
    ARTMAP_MODULES,             # List of supervised ARTMAP modules
    ADAPTIVERESONANCE_MODULES,  # Combined list of ART and ARTMAP modules
    DDVFA_METHODS,              # DDVFA linkage methods
    ADAPTIVERESONANCE_VERSION,  # Version of the package
    MATCH_FUNCTIONS,            # All match functions as a list of symbols
    ACTIVATION_FUNCTIONS,       # All activation functions as a list of symbols
    UPDATE_FUNCTIONS,           # All update functions as a list of symbols

    # ARTSCENE filter functions
    artscene_filter     # Runs all of the ARTSCENE functions in one step, returning features

# -----------------------------------------------------------------------------
# PRECOMPILE
# -----------------------------------------------------------------------------

# Precompile any concrete-type function signatures
@precompile_signatures(AdaptiveResonance)

end
