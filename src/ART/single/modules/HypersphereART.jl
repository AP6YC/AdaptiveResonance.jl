# -----------------------------------------------------------------------------
# TYPES
# -----------------------------------------------------------------------------

"""
Hypersphere ART options.

`rho` controls vigilance, `alpha` category choice, and `beta` learning speed.
`r_bar` is the radial extent in normalized feature units; `nothing` selects
`sqrt(dim) / 2`, the radius of the normalized unit cube, when training starts. For custom feature
scales, choose an extent at least half the largest pairwise sample distance.

$(_OPTS_DOCSTRING)
"""
@with_kw mutable struct opts_HypersphereART <: ARTOpts @deftype Float
    """Vigilance parameter in [0, 1]."""
    rho = 0.6; @assert 0.0 <= rho <= 1.0

    """Finite, positive choice parameter."""
    alpha = 1e-3; @assert isfinite(alpha) && alpha > 0.0

    """Learning rate in (0, 1]; 1 selects fast learning."""
    beta = 1.0; @assert 0.0 < beta <= 1.0

    """Radial extent, or `nothing` for an automatic dimension-based value."""
    r_bar::Union{Nothing,Float} = nothing
    @assert isnothing(r_bar) || (isfinite(r_bar) && r_bar > 0.0)

    """Maximum number of training epochs."""
    max_epoch::Int = 1; @assert max_epoch >= 1

    """Display progress bars."""
    display::Bool = false

    """Sort categories by activation before the vigilance search."""
    sort::Bool = false

    """Activation function for center-radius weights."""
    activation::Symbol = :hypersphere_activation

    """Match function for center-radius weights."""
    match::Symbol = :hypersphere_match

    """Update function for center-radius weights."""
    update::Symbol = :hypersphere_update
end

"""
    HypersphereART(; kwargs...)
    HypersphereART(opts::opts_HypersphereART)

Hypersphere ART learner supporting batch and incremental `train!` and `classify`,
including optional supervisory labels. Options are described in
[`opts_HypersphereART`](@ref).

Each column of `W` stores a center followed by its radius. Inputs are normalized
using `DataConfig` without complement coding. With `preprocessed=true`, inputs
are already normalized feature vectors, with no appended complements.
For incremental raw input, first call `data_setup!` or assign a `DataConfig`.

# References
G. C. Anagnostopoulos and M. Georgiopoulos (2000), "Hypersphere ART and ARTMAP
for unsupervised and supervised, incremental learning," IJCNN, vol. 6, pp. 59–64.
https://www.eecs.ucf.edu/georgiopoulos/sites/default/files/235.pdf
"""
mutable struct HypersphereART <: SingleART
    """Learner options."""
    opts::opts_HypersphereART

    """Feature normalization configuration."""
    config::DataConfig

    """Operating vigilance threshold."""
    threshold::Float

    """Effective radial extent, resolved when training starts."""
    r_bar::Float

    """Category labels."""
    labels::ARTVector{Int}

    """Category activation values."""
    T::ARTVector{Float}

    """Category match values."""
    M::ARTVector{Float}

    """Category centers and radii, stored as columns [center; radius]."""
    W::ARTMatrix{Float}

    """Number of samples assigned to each category."""
    n_instance::ARTVector{Int}

    """Number of committed categories."""
    n_categories::Int

    """Current training epoch."""
    epoch::Int

    """Statistics of the latest category search."""
    stats::ARTStats
end

# -----------------------------------------------------------------------------
# CONSTRUCTORS
# -----------------------------------------------------------------------------

# Build validated options from keywords and delegate to the options constructor.
HypersphereART(; kwargs...) = HypersphereART(opts_HypersphereART(; kwargs...))

function HypersphereART(opts::opts_HypersphereART)
    # Construct an empty learner; feature dimensions and radial extent are set later.
    return HypersphereART(
        opts,
        DataConfig(),
        opts.rho,
        0.0,
        Int[],
        Float[],
        Float[],
        ARTMatrix{Float}(undef, 0, 0),
        Int[],
        0,
        0,
        build_art_stats()
        )
end

# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------

# Hypersphere ART uses the original feature space, without complement coding.
function init_train!(x::RealVector, art::HypersphereART, preprocessed::Bool)
    # A single raw sample requires bounds supplied by the caller.
    if !art.config.setup
        preprocessed || error("HypersphereART: cannot preprocess data before being setup.")
        # Prepared samples supply the original feature dimension directly.
        art.config = DataConfig(0, 1, length(x))
    end
    # Validate the input and normalize only when it has not been prepared.
    return hypersphere_preprocess(x, art, preprocessed)
end

function init_train!(x::RealMatrix, art::HypersphereART, preprocessed::Bool)
    # Infer the normalization configuration only on the first training batch.
    if !art.config.setup
        if preprocessed
            # Prepared data already uses normalized feature coordinates.
            art.config = DataConfig(0, 1, size(x, 1))
        else
            # Raw batches provide the minimum and maximum of each feature.
            data_setup!(art, x)
        end
    end
    # Validate the input and normalize only when it has not been prepared.
    return hypersphere_preprocess(x, art, preprocessed)
end

function init_classify!(x::RealArray, art::HypersphereART, preprocessed::Bool)
    # Inference requires at least one committed category.
    art.n_categories > 0 || error("HypersphereART: cannot classify before training.")
    # Validate the input and normalize only when it has not been prepared.
    return hypersphere_preprocess(x, art, preprocessed)
end

"""Validate Hypersphere ART input dimensions and normalize original features."""
function hypersphere_preprocess(x::RealArray, art::HypersphereART, preprocessed::Bool)
    # Enforce the configured feature dimension and reject nonfinite samples.
    dim = art.config.dim
    dim > 0 || throw(ArgumentError("HypersphereART requires at least one feature."))
    size(x, 1) == dim || throw(DimensionMismatch("Expected $dim features, got $(size(x, 1))."))
    all(isfinite, x) || throw(ArgumentError("HypersphereART requires finite input values."))
    # Preserve prepared coordinates without normalizing or complement coding.
    preprocessed && return x
    # Check bounds before applying the shared normalization routine.
    all(art.config.mins .<= art.config.maxs) || throw(ArgumentError("Feature minima must not exceed maxima."))
    return linear_normalization(x, config=art.config)
end

function set_threshold!(art::HypersphereART)
    # Hypersphere match values use vigilance directly, without gamma scaling.
    art.threshold = art.opts.rho
end

function initialize!(art::HypersphereART, x::RealVector; y::Integer=0)
    # Initialize vigilance and resolve the radial extent in feature coordinates.
    set_threshold!(art)
    art.r_bar = isnothing(art.opts.r_bar) ? sqrt(art.config.dim) / 2 : art.opts.r_bar
    # Reserve one row per center coordinate and one final row for the radius.
    art.W = ARTMatrix{Float}(undef, art.config.dim + 1, 0)
    # Commit the first sample with its supervised label, or label 1.
    create_category!(art, x, iszero(y) ? 1 : y)
end

function create_category!(art::HypersphereART, x::RealVector, y::Integer)
    # Fast commit a sphere centered on the sample with zero initial radius.
    append!(art.W, vcat(x, 0.0))
    # Record the category label and its first assigned instance.
    push!(art.labels, y)
    push!(art.n_instance, 1)
    art.n_categories += 1
end

"""Search committed hyperspheres, preserving activation values for statistics."""
function hypersphere_search(art::HypersphereART; y::Integer=0)
    # Either sort once or keep a working copy for iterative winner inhibition.
    order = art.opts.sort ? sortperm(art.T, rev=true) : Int[]
    activations = art.opts.sort ? Float[] : copy(art.T)
    # Visit categories in descending activation order until one resonates.
    for j in 1:art.n_categories
        bmu = art.opts.sort ? order[j] : argmax(activations)
        # Accept only categories whose match reaches the vigilance threshold.
        if art.M[bmu] >= art.threshold
            # Follow the package's simple supervisory mismatch convention.
            !iszero(y) && art.labels[bmu] != y && return 0
            return bmu
        end
        # Activations can be zero or negative for distant samples.
        !art.opts.sort && (activations[bmu] = -Inf)
    end
    # Zero signals that no committed category passed the search.
    return 0
end

# COMMON DOC: HypersphereART incremental training method
function train!(art::HypersphereART, x::RealVector; y::Integer=0, preprocessed::Bool=false)
    # Set up the feature configuration and prepare the incoming sample.
    sample = init_train!(x, art, preprocessed)
    # Initialize the first category without searching an empty weight matrix.
    if iszero(art.n_categories)
        label = iszero(y) ? 1 : y
        initialize!(art, sample, y=label)
        return label
    end
    # An unseen supervisory label always starts a new category.
    if !iszero(y) && !(y in art.labels)
        create_category!(art, sample, y)
        return y
    end
    # Refresh vigilance and evaluate all categories through the shared symbols.
    set_threshold!(art)
    activation_match!(art, sample)
    # Retain the top activation for statistics and any mismatch fallback.
    top_bmu = argmax(art.T)
    bmu = hypersphere_search(art, y=y)
    mismatch = iszero(bmu)
    if mismatch
        # Commit a new sphere when the search finds no compatible category.
        label = iszero(y) ? art.n_categories + 1 : y
        create_category!(art, sample, label)
        bmu = top_bmu
    else
        # Update the resonant sphere and count the sample assigned to it.
        learn!(art, sample, bmu)
        art.n_instance[bmu] += 1
        label = art.labels[bmu]
    end
    # Store the winning activation, match, and mismatch status.
    log_art_stats!(art, bmu, mismatch)
    return label
end

# COMMON DOC: HypersphereART incremental classification method
function classify(art::HypersphereART, x::RealVector; preprocessed::Bool=false, get_bmu::Bool=false)
    # Prepare inference data using the normalization bounds learned in training.
    sample = init_classify!(x, art, preprocessed)
    # Refresh vigilance and evaluate all categories through the shared symbols.
    set_threshold!(art)
    activation_match!(art, sample)
    # Retain the top activation for statistics and any mismatch fallback.
    top_bmu = argmax(art.T)
    bmu = hypersphere_search(art)
    mismatch = iszero(bmu)
    # On mismatch, keep the strongest category for logging and optional fallback.
    mismatch && (bmu = top_bmu)
    # Store the winning activation, match, and mismatch status.
    log_art_stats!(art, bmu, mismatch)
    # Return -1 for rejection unless the caller requests the best available label.
    return mismatch && !get_bmu ? -1 : art.labels[bmu]
end
