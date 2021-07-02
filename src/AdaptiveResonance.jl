module AdaptiveResonance

# Usings/imports for the whole package declared once
using Parameters    # ARTopts are parameters (@with_kw)
using Logging       # Logging utils used as main method of terminal reporting
using ProgressBars  # Provides progress bars for training and inference
using Printf        # Used for printing formatted progress bars
using LinearAlgebra: norm   # Trace and norms
using Statistics: median, mean  # Medians and mean for linkage methods

# Abstract types
abstract type ARTOpts end   # ART module options
abstract type ART end       # ART modules

# Include all files
include("common.jl")        # Objects shared by all modules
include("constants.jl")     # Global constants and references for convenience
include("ARTMAP/ARTMAP.jl") # Supervised ART modules
include("ART/ART.jl")       # Unsupervised ART modules

# Export all public names
export

    # Abstract types
    ARTOpts,        # All module options are ARTOpts
    ART,            # All modules are ART modules

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
    DDVFA, opts_DDVFA,
    GNFA, opts_GNFA,
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
