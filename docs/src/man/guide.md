# Package Guide

The `AdaptiveResonance.jl` package is built upon ART modules that contain all of the state information during training and inference.
The ART modules are driven by options, which are themselves mutable keyword argument structs from the [Parameters.jl](https://github.com/mauro3/Parameters.jl) package.

To work with `AdaptiveResonance.jl`, you should know:

- [How to install the package](@ref installation)
- [ART module basics](@ref art_modules)
- [How to use ART module options](@ref art_options)
- [ART vs. ARTMAP](@ref art_vs_artmap)

## [Installation](@id installation)

The AdaptiveResonance package can be installed using the Julia package manager.
From the Julia REPL, type `]` to enter the Pkg REPL mode and run

```julia
pkg> add AdaptiveResonance
```

Alternatively, it can be added to your environment in a script with

```julia
using Pkg
Pkg.add("AdaptiveResonance")
```

If you wish to have the latest changes between releases, you can directly add the GitHub repo as a dependency with

```julia
pkg> add https://github.com/AP6YC/AdaptiveResonance.jl
```

## [ART Modules](@id art_modules)

To work with ART modules, you should know:

- [Their basic methods](@ref methods)
- [Incremental vs. batch modes](@ref incremental_vs_batch)
- [Supervised vs. unsupervised learning modes](@ref supervised_vs_unsupervised)
- [Mismatch vs. Best-Matching-Unit](@ref mismatch-bmu)

### [Methods](@id methods)

Every ART module is equipped with several constructors, a training function `train!`, and a classification/inference function `classify`.
ART models are mutable structs, and they can be instantiated with

```julia
art = DDVFA()
```

For more ways to customize instantiation, see the [ART options section](@ref art_options).

To train and test these models, you use the `train!` and `classify` functions upon the models.
Because training changes the internal parameters of the ART models and classification does not, `train!` uses an exclamation point while `classify` does not, following Julia standard usage.

For example, we may load data of some sort and train/test like so:

```julia
# Load the data from some source with a train/test split
train_x, train_y, test_x, test_y = load_some_data()

# Instantiate an arbitrary ART module
art = DDVFA()

# Train the module on the training data, getting the prescribed cluster labels
y_hat_train = train!(art, train_x)

# Conduct inference
y_hat_test = classify(art, test_x)
```

!!! note "Note"
    Because Julia arrays are column-major in memory, the `AdaptiveResonance.jl` package follows the Julia convention of assuming 2-D data arrays are in the shape of `(n_features, n_samples)`.

### [Incremental vs. Batch](@id incremental_vs_batch)

This training and testing may be done in either incremental or batch modes:

```julia
# Create a destination container for the incremental examples
n_train = length(train_y)
n_test = length(test_y)
y_hat_train_incremental = zeros(Integer, n_train)
y_hat_test_incremental = zeros(Integer, n_test)

# Loop over all training samples
for i = 1:n_train
    y_hat_train_incremental[i] = train!(art, train_x[:, i])
end

# loop over all testing samples
for i = 1:n_test
    y_hat_test_incremental[i] = classify(art, test_x[:, i])
end
```

This is done through checking the dimensionality of the inputs.
For example, if a matrix (i.e., 2-D array) is passed to the `train!` function, then the data is assumed to be `(n_features, n_samples)`, and the module is trained on all samples.
However, if the data is a vector (i.e., 1-D array), then the vector is interpreted as a single sample.

When supervised (see [supervised vs. unsupervised](@ref supervised_vs_unsupervised)), the dimensions of the labels must correspond to the dimensions of the data.
For example, a 2-D matrix of the data must accompany a 1-D vector of labels, while a 1-D vector of a single data sample must accompany a single integer label.

Batch and incremental modes can be used interchangably after module instantiation.

!!! note "Note"
    The first time that an ART module is trained, it infers the data parameters (e.g., feature dimensions, feature ranges, etc.) to setup the internal data configuration.
    This happens automatically in batch mode, but it cannot happen if the module is only trained incrementally.
    If you know the dimensions and minimum/maximum values of the features and want to train incrementally, you can use the function `data_setup!` after module instantiation, which can be used a number of ways.
    If you have the batch data available, you can set up with

    ```julia
    # Manually setup the data config with the data itself
    data_setup!(art.config, data.train_x)
    ```

    If you do not have the batch data available, you can directly create a `DataConfig` with the minimums and maximums (inferring the number of features from the lengths of these vectors):

    ```julia
    # Get the mins and maxes vectors with some method
    mins, maxes = get_some_data_mins_maxes()

    # Directly update the data config
    art.config = DataConfig(mins, maxes)
    ```

    If all of the features share the same minimums and maximums, then you can use them as long as you specify the number of features:

    ```julia
    # Get the global minimum, maximum, and feature dimension somehow
    min, max, dim = get_some_data_characteristics()

    # Directly update the data config with these global values
    art.config = DataConfig(min, max, dim)
    ```

### [Supervised vs. Unsupervised](@id supervised_vs_unsupervised)

ARTMAP modules require a supervised label argument because their formulations typically map internal cluster categories to labels:

```julia
# Create an arbitrary ARTMAP module
artmap = DAM()

# Conduct supervised learning
y_hat_train = train!(artmap, train_x, train_y)

# Conduct inference
y_hat_test = classify(artmap, test_x)
```

In the case of ARTMAP, the returned training labels `y_hat_train` will always match the training labels `train_y` by definition.
In addition to the classification accuracy (ranging from 0 to 1), you can test that the training labels match with the function `performance`:

```julia
# Verify that the training labels match
perf_train = performance(y_hat_train, train_y)

# Get the classification accuracy
perf_test = performance(y_hat_test, test_y)
```

However, many ART modules, though unsupervised by definition, can also be trained in a supervised way by naively mapping categories to labels (more in [ART vs. ARTMAP](@ref art_vs_artmap)).

### [Mismatch vs. Best-Matching-Unit](@id mismatch-bmu)

During inference, ART algorithms report the category that satisfies the match/vigilance criterion (see [Background](@ref)).
By default, in the case that no category satisfies this criterion the module reports a *mismatch* as -1.
In modules that support it, a keyword argument `get_bmu` (default is `false`) can be used in the `classify` method to get the "best-matching unit", which is the category that maximizes the activation.
This can be interpreted as the "next-best guess" of the model in the case that the sample is sufficiently different from anything that the model has seen.
For example,

```julia
# Conduct inference, getting the best-matching unit in case of complete mismatch
y_hat_bmu = classify(my_art, test_x, get_bmu=true)
```

## [ART Options](@id art_options)

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

The options are objects from the [Parameters.jl](https://github.com/mauro3/Parameters.jl) project, so they can be instantiated even with keyword arguments:

```julia
my_art_opts = opts_DDVFA(gamma = 3)
```

!!! note "Note"
    As of version `0.3.6`, you can pass these keyword arguments directly to the ART model when constructing it with

    ```julia
    my_art = DDVFA(gamma = 3)
    ```

You can even modify the parameters on the fly after the ART module has been instantiated by directly modifying the options within the module:

```julia
my_art = DDVFA()
my_art.opts.gamma = 3
```

Because of the `@assert` feature of the Parameters.jl package, each parameter is forced to lie within certain bounds by definition in the literature during options instantiation.
However, it is possible to change these parameter values beyond their predefined bounds after instantiation.

!!! note "Note"
    You must be careful when changing option values during or after training, as it may result in some undefined behavior.
    Modify the ART module options after instantiation at your own risk and discretion.

Though most parameters differ between each ART and ARTMAP module, they all share some quality-of-life options and parameters shared by all ART algorithms:

- `display::Bool`: a flag to display or suppress progress bars and logging messages during training and testing.
- `max_epochs::Integer`: the maximum number of epochs to train over the data, regardless if other stopping conditions have not been met yet.

Otherwise, most ART and ARTMAP modules share the following nomenclature for algorithmic parameters:

- `rho::Float`: ART vigilance parameter [0, 1].
- `alpha::Float`: Choice parameter > 0.
- `beta::Float`: Learning parameter (0, 1].
- `epsilon::Float`: Match tracking parameter (0, 1).

## [ART vs. ARTMAP](@id art_vs_artmap)

ART modules are generally unsupervised in formulation, so they do not explicitly require supervisory labels to their training examples.
However, many of these modules can be formulated in the simplified ARTMAP style whereby the ART B module has a vigilance parameter of 1, directly mapping the categories of the ART A module to any provided supervisory labels.

This is done in the training stage through the optional argument `y=...`:

```julia
# Create an arbitrary ART module
art = DDVFA()

# Naively prescribe supervised labels to cluster categories
y_hat_train = train!(art, train_x, y=train_y)
```

This can also be done incrementally with the same function:

```julia
# Get the number of training samples and create a results container
n_train = length(train_y)
y_hat_train_incremental = zeros(Integer, n_train)

# Train incrementally over all training samples
for i = 1:n_train
    y_hat_train_incremental[i] = train!(art, train_x[:, i], y=train_y[i])
end
```

Without provided labels, the ART modules behave as expected, incrementally creating categories when necessary during the training phase.
