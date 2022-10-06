using AdaptiveResonance

# Create a FuzzyART module with default options
my_fuzzyart = FuzzyART()
typeof(my_fuzzyart)

# Check the FuzzyART options
my_fuzzyart.opts

# Create a DDVFA module and check the type of the options
my_ddvfa = DDVFA()
typeof(my_ddvfa.opts)

# Create a separate options struct
my_fuzzyart_opts = opts_FuzzyART()

# Instantiate an ART module by passing our options
my_fuzzyart = FuzzyART(my_fuzzyart_opts)
my_other_fuzzyart = FuzzyART(my_fuzzyart_opts)

# Change some of the default FuzzyART options
my_fuzzyart_opts = opts_FuzzyART(
    rho=0.6,
    gamma_normalization=true
)
my_fuzzyart = FuzzyART(my_fuzzyart_opts)

# Pass these keyword arguments to the module directly
my_fuzzyart = FuzzyART(
    rho=0.6,
    gamma_normalization=true
)

my_fuzzyart = FuzzyART()
my_fuzzyart.opts.rho=0.6

using MLDatasets        # Iris dataset
using DataFrames        # DataFrames, necessary for MLDatasets.Iris()
using MLDataUtils       # Shuffling and splitting
using Printf            # Formatted number printing
using MultivariateStats # Principal component analysis (PCA)
using Plots             # Plotting frontend
gr()                    # Use the default GR backend explicitly

# Get the iris dataset
iris = Iris(as_df=false)
# Manipulate the features and labels into a matrix of features and a vector of labels
features, labels = iris.features, iris.targets

labels = convertlabel(LabelEnc.Indices{Int}, vec(labels))
unique(labels)

(X_train, y_train), (X_test, y_test) = stratifiedobs((features, labels))

# Create two FuzzyARTs with different vigilance values and suppressing logging messages
rho_1 = 0.5
rho_2 = 0.7
my_fuzzyart_1 = FuzzyART(rho=rho_1, display=false)
my_fuzzyart_2 = FuzzyART(rho=rho_2, display=false)

# Train in simple supervised mode by passing the labels as a keyword argument.
y_hat_train_1 = train!(my_fuzzyart_1, X_train, y=y_train)
y_hat_train_2 = train!(my_fuzzyart_2, X_train, y=y_train)

y_hat_1 = AdaptiveResonance.classify(my_fuzzyart_1, X_test, get_bmu=true)
y_hat_2 = AdaptiveResonance.classify(my_fuzzyart_2, X_test, get_bmu=true)

# Check the shape and type of the output labels
println("FuzzyART 1 labels: ",  size(y_hat_1), " ", typeof(y_hat_1))
println("FuzzyART 2 labels: ",  size(y_hat_2), " ", typeof(y_hat_2))

# Calculate the performance on the test data
perf_test_1 = performance(y_hat_1, y_test)
perf_test_2 = performance(y_hat_2, y_test)

# Format each performance number for comparison
@printf "Testing performance rho=%.1f: %.4f\n" rho_1 perf_test_1
@printf "Testing performance rho=%.1f: %.4f\n" rho_2 perf_test_2

# Print the number of categories for each vigilance parameter
@printf "Number of categories rho=%.1f: %i\n" rho_1 my_fuzzyart_1.n_categories
@printf "Number of categories rho=%.1f: %i\n" rho_2 my_fuzzyart_2.n_categories

# Train a PCA model to visually separate the features in two dimensions.
M = fit(PCA, features; maxoutdim=2)

# Apply the PCA model to the testing set
X_test_pca = MultivariateStats.transform(M, X_test)

# Create a function for our subplots
function fuzzyart_scatter(data, labels, rho)
    p = scatter(
        data[1, :],             # PCA dimension 1
        data[2, :],             # PCA dimension 2
        group=labels,           # labels belonging to each point
        markersize=8,           # size of scatter points
        xlims = [-4, 4],        # manually set the x-limits
        title=(@sprintf "FuzzyART \$\\rho\$ = %.1f" rho),  # formatted title
    )
    return p
end

# Create the two scatterplot objects
p1 = fuzzyart_scatter(X_test_pca, y_hat_1, rho_1)
p2 = fuzzyart_scatter(X_test_pca, y_hat_2, rho_2)

# Plot the two scatterplots together
plot(
    p1, p2,                 # scatterplot objects
    layout = (1, 2),        # plot side-by-side
    ##layout = [a, b],        # plot side-by-side
    legend = false,         # no legend
    xtickfontsize = 12,     # x-tick size
    ytickfontsize = 12,     # y-tick size
    xlabel = "\$PCA_1\$",   # x-label
    ylabel = "\$PCA_2\$",   # y-label
    dpi = 300,              # Set the dots-per-inch
)

png("assets/options-cover") #hide

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

