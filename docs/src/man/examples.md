# Examples

There are examples for every structure in the package within the package's ```examples/``` folder.
The code for several of these examples is provided here.

## ART

All ART modules learn in an unsupervised (i.e. clustering) mode by default, but they all can accept labels in the simplified ARTMAP fashion (see the [Package Guide](@ref)).

### DDVFA Unsupervised

```julia
# Load the data from some source with a train/test split
train_x, train_y, test_x, test_y = load_some_data()

# Instantiate a DDVFA module
art = DDVFA()

# Train the module on the training data, getting the prescribed cluster labels
y_hat_train = train!(art, train_x)

# Conduct inference
y_hat_test = classify(art, test_x)
```

### DDVFA Supervised

```julia
# Load the data from some source with a train/test split
train_x, train_y, test_x, test_y = load_some_data()

# Instantiate a DDVFA module
art = DDVFA()

# Train the module on the training data, getting the prescribed cluster labels
y_hat_train = train!(art, train_x, y=train_y)

# Conduct inference
y_hat_test = classify(art, test_x)

# Verify that the training labels match
perf_train = performance(y_hat_train, train_y)

# Get the classification accuracy
perf_test = performance(y_hat_test, test_y)
```

### Incremental DDVFA With Custom Options and Data Configuration

```julia
# Load the data from some source with a train/test split
train_x, train_y, test_x, test_y = load_some_data()

# Create custom DDVFA options
opts = opts_DDVFA(gamma=3)

# Instantiate a DDVFA module with the specified options
art = DDVFA(opts)

# Change the options after instantiation for fun
art.opts.rho_lb = 0.5

# Customize the data configuration
# Assume that we have prior knowledge that the features lie within [0, 1]
# and that they have dimension 10
art.config = DataConfig(0, 1, 10)

# Create data containers for label results
n_train = length(train_x)
n_test = length(test_x)
y_hat_train = zeros(Integer, n_train)
y_hat_test = zeros(Integer, n_test)

# Train the module on the training data incrementally, getting the prescribed cluster labels
for i = 1:n_train
    sample = train_x[:, i]
    label = train_y[i]
    y_hat_train[i] = train!(art, sample, y=label)
end

# Conduct inference incrementally
for i = 1:n_test
    sample = test_x[:, i]
    y_hat_test[i] = classify(art, sample)
end

# Verify that the training labels match
perf_train = performance(y_hat_train, train_y)

# Get the classification accuracy
perf_test = performance(y_hat_test, test_y)
```

## ARTMAP

ARTMAP modules are supervised by definition, so the require supervised labels in the training stage.

### SFAM

```julia
# Load the data from some source with a train/test split
train_x, train_y, test_x, test_y = load_some_data()

# Create a Simplified Fuzzy ARTMAP module
art = SFAM()

# Train in batch
y_hat_train = train!(art, train_x, train_y)

# Test in batch
y_hat_test = classify(art, test_x)

# Verify that the training labels match
perf_train = performance(y_hat_train, train_y)

# Calculate testing performance
perf_test = performance(y_hat_test, test_y)
```
### Incremental SFAM With Custom Options and Data Configuration

```julia
# Load the data from some source with a train/test split
train_x, train_y, test_x, test_y = load_some_data()

# Create custom SFAM options
opts = opts_SFAM(rho=0.5)

# Instantiate a SFAM module with the specified options
art = SFAM(opts)

# Change the options after instantiation for fun
art.opts.epsilon = 1e-2

# Customize the data configuration
# Assume that we have prior knowledge that the features lie within [0, 1]
# and that they have dimension 10
art.config = DataConfig(0, 1, 10)

# Create data containers for label results
n_train = length(train_x)
n_test = length(test_x)
y_hat_train = zeros(Integer, n_train)
y_hat_test = zeros(Integer, n_test)

# Train the module on the training data incrementally, getting the prescribed cluster labels
for i = 1:n_train
    sample = train_x[:, i]
    label = train_y[i]
    y_hat_train[i] = train!(art, sample, label)
end

# Conduct inference incrementally
for i = 1:n_test
    sample = test_x[:, i]
    y_hat_test[i] = classify(art, sample)
end

# Verify that the training labels match
perf_train = performance(y_hat_train, train_y)

# Get the classification accuracy
perf_test = performance(y_hat_test, test_y)
```
