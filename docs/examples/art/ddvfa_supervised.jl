# ---
# title: Supervised DDVFA Example
# id: ddvfa_supervised
# cover: ../assets/ddvfa.png
# date: 2021-11-30
# author: "[Sasha Petrenko](https://github.com/AP6YC)"
# julia: 1.6
# description: This demo shows how to use DDVFA for simple supervised learning by clustering Iris samples and mapping the modules internal categories to the true labels.
# ---

# DDVFA is an unsupervised clustering algorithm by definition, but it can be adaptived for supervised learning by mapping the module's internal categories to the true labels.

# We begin with importing AdaptiveResonance for the ART modules and MLDatasets for some data utilities.
using AdaptiveResonance
using MLDatasets
using MLDataUtils
using Logging

# Set the log level
LogLevel(Logging.Info)

# We will download the Iris dataset for its small size and benchmark use for clustering algorithms.
Iris.download(i_accept_the_terms_of_use=true)
features, labels = Iris.features(), Iris.labels()

(X_train, y_train), (X_test, y_test) = stratifiedobs((features, labels))

# Train and classify
art = DDVFA()
y_hat_train = train!(art, X_train, y=y_train)
y_hat = classify(art, X_test)
y_hat_bmu = classify(art, X_test, get_bmu=true)

# Calculate performance
perf_train = performance(y_hat_train, y_train)
perf_test = performance(y_hat, y_test)
perf_test_bmu = performance(y_hat_bmu, y_test)

@info "DDVFA Training Perf: $perf_train"
@info "DDVFA Testing Perf: $perf_test"
@info "DDVFA Testing BMU Perf: $perf_test_bmu"
