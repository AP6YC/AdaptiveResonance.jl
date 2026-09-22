"""
Evaluate the resonance search for a module.

# Arguments
- `accept::F`: anonymous function to pass the accept/reject decision to the ART module.
- `art::ARTModule`: the ART module running the resonance search.
- `sample::RealVector`: the sample presented for search.
- `threshold=art.threshold`: the vigilance threshold (can be rho or a function that varies during training/evaluation). Default `art.threshold`

# Description

Evaluates a preprocessed sample against a nonempty module and return `(bmu, mismatch)`.
On mismatch, `bmu` is the original maximum-activation category.
Records its activation, match, and mismatch status in `art.stats`.

Candidates are visited in descending activation order, either by sorting or
by inhibiting a working copy with `-Inf` (determined by the `art.opts.sort` flag).
The original activations are retained for statistics, including when no category resonates.

`accept(bmu)` is called only when vigilance passes: return `true` to accept,
`false` to stop with a supervisory mismatch, or `nothing` to continue searching
(e.g. after ARTMAP match tracking).
`threshold` may be a number or a zero-argument function for a changing vigilance threshold to recompute it (such as in vigilance tracking).
The default callback accepts any label when `y == 0`, otherwise it stops on a conflicting label.

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
    threshold=art.threshold
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

    # Loop flag
    mismatch = true

    # Loop over all categories
    for j in 1:art.n_categories
        # Get the next candidate
        candidate = art.opts.sort ? order[j] : argmax(activations)

        # Compute the match value
        match = resonance_match!(art, sample, candidate)

        # Get the threshold (sometimes rho, sometimes a function of rho, etc.)
        vigilance = threshold isa Number ? threshold : threshold()

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
                break
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
    y::Integer=0
)
    return resonance_search!(art, sample; threshold=threshold) do bmu
        iszero(y) || art.labels[bmu] == y
    end
end

# Temporary workaround and placeholder for more elegant activation and match handling
resonance_activation!(art::ARTModule, sample::RealVector) = activation_match!(art, sample)
resonance_match!(art::ARTModule, sample::RealVector, bmu::Integer) = art.M[bmu]
