"""
Main module for `AdaptiveResonance.jl`, a Julia package of adaptive resonance theory algorithms.

This module exports all of the ART modules, options, and utilities used by the `AdaptiveResonance.jl package.`

# Exports

$(EXPORTS)

"""
module AdaptiveResonance

# --------------------------------------------------------------------------- #
# USINGS
# --------------------------------------------------------------------------- #

# Usings/imports for the whole package declared once
using DocStringExtensions   # Docstring utilities
using Parameters    # ARTopts are parameters (@with_kw)
using Logging       # Logging utils used as main method of terminal reporting
using ProgressBars  # Provides progress bars for training and inference
using Printf        # Used for printing formatted progress bars
using LinearAlgebra: norm   # Trace and norms
using Statistics: median, mean  # Medians and mean for linkage methods

# --------------------------------------------------------------------------- #
# ABSTRACT TYPES
# --------------------------------------------------------------------------- #

"""
    ARTOpts

Abstract supertype for all ART module options.
"""
abstract type ARTOpts end               # ART module options

"""
    ARTModule

Abstract supertype for both ART (unsupervised) and ARTMAP (supervised) modules.
"""
abstract type ARTModule end             # ART modules

"""
    ART <: ARTModule

Abstract supertype for all default unsupervised ART modules.
"""
abstract type ART <: ARTModule end      # ART (unsupervised)

"""
    ARTMAP <: ARTModule

Abstract supertype for all supervised ARTMAP modules.
"""
abstract type ARTMAP <: ARTModule end   # ARTMAP (supervised)

# --------------------------------------------------------------------------- #
# INCLUDES
# --------------------------------------------------------------------------- #

# Include all files
include("common.jl")        # Objects shared by all modules
include("constants.jl")     # Global constants and references for convenience
include("ARTMAP/ARTMAP.jl") # Supervised ART modules
include("ART/ART.jl")       # Unsupervised ART modules

# --------------------------------------------------------------------------- #
# EXPORTS
# --------------------------------------------------------------------------- #

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
    get_data_shape,             # Get the dim, n_samples of x (accepts 1-D and 2-D)
    get_n_samples,              # Get the number of samples (1-D interpreted as one sample)

    # ART (unsupervised)
    FuzzyART, opts_FuzzyART,
    DDVFA, opts_DDVFA, get_W,
    DVFA, opts_DVFA,

    # ARTMAP (supervised)
    FAM, opts_FAM,
    DAM, opts_DAM,
    SFAM, opts_SFAM,

    # ARTSCENE filter functions
    color_to_gray,
    contrast_normalization,
    contrast_sensitive_oriented_filtering,
    contrast_insensitive_oriented_filtering,
    orientation_competition,
    patch_orientation_color,
    artscene_filter     # Runs all of the above in one step, returning features

end # module
