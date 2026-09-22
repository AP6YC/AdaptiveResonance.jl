"""
Evaluate the resonance search for a module.

# Arguments
- `accept::F`: anonymous function to pass the accept/reject decision to the ART module.
- `art::ARTModule`: the ART module running the resonance search.
- `sample::RealVector`: the sample presented for search.
- `match_tracking=art.opts.match_tracking`: continue searching with raised vigilance after a callback rejects a label.
- `threshold=art.threshold`: the vigilance threshold (can be rho or a function that varies during training/evaluation). Default `art.threshold`

# Description

Evaluates a preprocessed sample against a nonempty module and return `(bmu, mismatch)`.
On mismatch, `bmu` is the original maximum-activation category.
Records its activation, match, and mismatch status in `art.stats`.

Candidates are visited in descending activation order, either by sorting or
by inhibiting a working copy with `-Inf` (determined by the `art.opts.sort` flag).
The original activations are retained for statistics, including when no category resonates.

`accept(bmu)` is called only when vigilance passes: return `true` to accept,
`false` to signal a supervisory mismatch, or `nothing` to continue searching.
On `false`, `match_tracking=true` raises the local vigilance above the rejected
match by `opts.epsilon` (scaled to match units) and continues; otherwise search
stops. The baseline is never mutated, and tracked vigilance is not capped, so
conflicting exact matches can exhaust the search and request a new category.
`threshold` may be a number or a zero-argument function for a changing vigilance threshold to recompute it (such as in vigilance tracking).
The default callback checks labels only when `supervised=true` (default: `y != 0`).
ARTMAP callers explicitly enable supervision so label zero is also supported.

# Examples

```julia
resonance_search!(art, sample; threshold=art.threshold, y=0)
resonance_search!(accept, art, sample; threshold=art.threshold)
```
"""
function resonance_search!(
    accept::F,
    art::ARTModule,
    sample::RealVector;
    threshold=art.threshold,
    match_tracking::Bool=art.opts.match_tracking
) where {F}
    # Error if doing a resonance search without any categories
    art.n_categories > 0 || throw(ArgumentError("Resonance search requires a committed category."))

    # Compute the activation
    resonance_activation!(art, sample)

    # If using the presort strategy, do so in advance
    order = art.opts.sort ? sortperm(art.T, rev=true) : Int[]

    # If presorted, just use T; otherwise, copy for inhibiting entries
    activations = art.opts.sort ? art.T : copy(art.T)

    # Start with highest activation to begin search (best matching unit)
    bmu = art.opts.sort ? first(order) : argmax(activations)

    # Tracking raises only a local floor, never the stored baseline vigilance.
    # This also permits a caller-supplied threshold function to vary independently.
    tracked_vigilance = -Inf

    # Loop flag
    mismatch = true

    # Loop over all categories
    for j in 1:art.n_categories
        # Get the next candidate
        candidate = art.opts.sort ? order[j] : argmax(activations)

        # Compute the match value
        match = resonance_match!(art, sample, candidate)

        # Get the threshold (sometimes rho, sometimes a function of rho, etc.)
        baseline = threshold isa Number ? threshold : threshold()
        vigilance = max(baseline, tracked_vigilance)

        # Vigilance test
        if match >= vigilance
            # Handle bmu acceptance outside this function, dependent on module
            decision = accept(candidate)
            # If accepted, return bmu
            if decision === true
                bmu = candidate
                mismatch = false
                break
            elseif decision === false
                # A label conflict either ends simple supervised search or raises
                # vigilance just above this match and searches the next category.
                if !match_tracking
                    break
                end
                increment = art.opts.epsilon * resonance_match_scale(art)
                # nextfloat guarantees a strict increase even if epsilon rounds away.
                tracked_vigilance = max(match + increment, nextfloat(float(match)))
            end
        end
        # If not presorted and no match, manually inhibit the candidate with -Inf
        !art.opts.sort && (activations[candidate] = -Inf)
    end

    # Log the activation and match
    log_art_stats!(art, bmu, mismatch)

    # Return the bmu and match result
    return bmu, mismatch
end

# Wrapper for resonance search, handling supervised and unsupervised cases
function resonance_search!(
    art::ARTModule,
    sample::RealVector;
    threshold=art.threshold,
    y::Integer=0,
    supervised::Bool=!iszero(y)
)
    # Only supervised searches can produce the prediction error needed for tracking.
    return resonance_search!(art, sample;
                             threshold=threshold,
                             match_tracking=supervised && art.opts.match_tracking) do bmu
        !supervised || art.labels[bmu] == y
    end
end

"""
Prepare category activations and match storage before resonance search.

# Arguments
- `art::ARTModule`: the ART module running the resonance search.
- `sample::RealVector`: the preprocessed sample presented for search.

# Description

The default delegates to [`activation_match!`](@ref). SFAM prepares activations
and defers matches until candidates are visited; DDVFA first evaluates its local
modules and computes global activations using the configured linkage.
"""
resonance_activation!(art::ARTModule, sample::RealVector) = activation_match!(art, sample)

"""
Return the candidate's match value during resonance search.

# Arguments
- `art::ARTModule`: the ART module running the resonance search.
- `sample::RealVector`: the preprocessed sample presented for search.
- `bmu::Integer`: the index of the candidate category being evaluated.

# Description

The default reads
`art.M[bmu]`; SFAM and DDVFA compute and store this value lazily for each visited
candidate, using their match function or linkage respectively.
"""
resonance_match!(art::ARTModule, sample::RealVector, bmu::Integer) = art.M[bmu]

"""
Return the scale for converting a vigilance-parameter increment to match units.

# Arguments
- `art::ARTModule`: the ART module whose match scale determines the tracking increment.

# Description

Resonance search raises its temporary threshold by
`art.opts.epsilon * resonance_match_scale(art)`.
The default scale is one. DVFA uses the feature dimension; FuzzyART and DDVFA
use `dim ^ gamma_ref` when gamma normalization is enabled, and one otherwise.
"""
resonance_match_scale(::ARTModule) = 1.0
