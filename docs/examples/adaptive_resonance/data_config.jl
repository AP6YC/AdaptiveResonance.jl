# ---
# title: ART DataConfig Example
# id: data_config
# cover: ../assets/art.png
# date: 2021-12-2
# author: "[Sasha Petrenko](https://github.com/AP6YC)"
# julia: 1.6
# description: This demo illustrates how the data configuration object works for data preprocessing in ART modules that require it.
# ---

# ## Overview

# In their derivations, ART modules have some special requirements when it comes to their input features.
# FuzzyART in particular, and subsequently its derivatives, has a requirement that the inputs be bounded and complement coded.
# This is due to some consequences such as weight decay that occur when using real-valued patterns rather than binary ones (and hence operations like fuzzy membership).

# Preprocessing of the features occurs as follows:
# 1. The features are linearly normalized from 0 to 1 with respect to each feature with `linear_normalization`.
#    This is done according to some known bounds that each feature has.
# 2. The features are then complement coded, meaning that the feature vector is appended to its 1-complement (i.e., $x \rightarrow \left[x, 1-x\right]$) with `complement_code`.

# This preprocessing has the ultimate consequence that the input features must be bounded.
# This many not be a problem in some offline applications with a fixed dataset, but in others where the bounds are not known, techniques such as sigmoidal limiting are often used to place an artificial limit.

# ## DataConfig

# Regardless, this process requires some *a-priori* knowledge about the minimums and maximums that each feature can have, which is stored as a preprocessing configuration.
# This preprocessing configuration is saved in every ART module as a `DataConfig` object called `config`, which we can see is uninitialized at first:

## Load the library
using AdaptiveResonance

## Create a new ART module and inspect its uninitialized data config `config`
art = FuzzyART()
art.config

# We see that the type of `art.config` is `DataConfig`.
# We can see what the internal elements of this struct are with `fieldnames`:
fieldnames(AdaptiveResonance.DataConfig)

# We see that the dataconfig has a boolean setup flag, minimum and maximum feature vectors, dimensionality of the data, and the complement coded dimensionality (twice the size of the original dimension).

# ### Automatic Configuration

# In batch training mode, the minimums and maximums are detected automatically; the minimum and maximum values for every feature are saved and used for the preprocessing step at every subsequent iteration.

## Load data
using MLDatasets
using MLUtils

## We will download the Iris dataset for its small size and benchmark use for clustering algorithms.
iris = Iris()
features, labels = Matrix(iris.features), Matrix{String}(iris.targets)

# Because the MLDatasets package gives us Iris labels as strings, we will use the `MLDataUtils.convertlabel` method with the `MLLabelUtils.LabelEnc.Indices` type to get a list of integers representing each class:
labels = convertlabel(LabelEnc.Indices{Int}, vec(labels))
unique(labels)

# !!! note
#     This automatic detection of feature characteristics only occurs if the `config` is not already setup.
#     If it is setup beforehand, then that config is used instead.

# ### Manual Configuration

# As mentioned before, we may not always have the luxury of having a representative dataset in advance.
# Alternatively, we may know the bounds of the features but wish to run incrementally rather than in batch.
# In these cases, we can setup the config the various `DataConfig` constructors.

# For example, if the features are all bounded from -1 to 1, we have to also specify the original dimension of the data in `DataConfig(min, max, dim)`:

## Reinitialize the FuzzyART module
art = FuzzyART()
## Tell the module that we have 20 features all ranging from -1 to 1
art.config = DataConfig(-1, 1, 20)

# If the features differ in their ranges, we can specify with `DataConfig(mins, maxs)`:

## Assume some minimum and maximum values for each feature
mins = [-1,-2,-1.5]
maxs = [3, 2, 1]
art.config = DataConfig(mins, maxs)

# Here, we don't need to specify the feature dimensionality because it is inferred from the length of the range values.

# !!! note
#     After the first training run, the weights of the network are set to the size of the complement coded dimension.
#     If you wish to change the dimension of the features, you will need to create a new network.
