"""
    common_docs.jl

# Description
Common docstrings for definitions in the package, to be included after those definitions are defined.
"""

# -----------------------------------------------------------------------------
# COMMON DOCUMENTATION
# -----------------------------------------------------------------------------

@doc """
Predict categories of a single sample of features 'x' using the ART model.

Returns predicted category 'y_hat.'

# Arguments
- `art::ARTModule`: ART or ARTMAP module to use for batch inference.
- `x::RealVector`: the single sample of features to classify.
- `preprocessed::Bool=false`: optional, flag if the data has already been complement coded or not.
- `get_bmu::Bool=false`: optional, flag if the model should return the best-matching-unit label in the case of total mismatch.
"""
classify(art::ARTModule, x::RealVector ; preprocessed::Bool=false, get_bmu::Bool=false)

# Common function for setting the threshold (sometimes just vigilance, sometimes a function of vigilance).
@doc """
Sets the match threshold of the ART/ARTMAP module as a function of the vigilance parameter.

Depending on selected ART/ARTMAP module and its options, this may be a function of other parameters as well.

# Arguments
- `art::ARTModule`: the ART/ARTMAP module for setting a new threshold.
"""
set_threshold!(art::ARTModule)

@doc """
Creates a category for the ARTModule module, expanding the weights and incrementing the category labels.

# Arguments
- `art::ARTModule`: the ARTModule module to add a category to.
- `x::RealVector`: the sample to use for adding a category.
- `y::Integer`: the new label for the new category.
"""
create_category!(art::ARTModule, x::RealVector, y::Integer)

"""
Common docstring for listing available match functions.
"""
const MATCH_FUNCTIONS_DOCS = join(MATCH_FUNCTIONS, ", ", " and ")

"""
Common docstring for listing available activation functions.
"""
const ACTIVATION_FUNCTIONS_DOCS = join(ACTIVATION_FUNCTIONS, ", ", " and ")
