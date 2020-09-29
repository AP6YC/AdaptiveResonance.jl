using DataFrames
using DrWatson
using AdaptiveResonance
using Parameters
using CSV
using Logging
using ProgressMeter
using Distributed

@info "Loading data..."
data_rng = convert(Array, CSV.read(datadir("art_data_rng.csv")))
nSamples, dim = size(data_rng)
@info "Data loaded" data_rng

n_epochs = 1

opts_DDVFA_1 = opts_DDVFA()
# @info "Instantiating DDVFA with parameters" params_DDVFA_1
DDVFA_1 = DDVFA(opts=opts_DDVFA_1)
train!(DDVFA_1, data_rng, n_epochs)

GNFA_1 = GNFA()
train!(GNFA_1, data_rng, n_epochs)

# params_MFA_1 = params_MFA()
# @info "Instantiating MFA with parameters" params_MFA_1

nEpochs_MFA = 1
t = 1

@showprogress pmap(1:10) do x
    sleep(0.1)
    x^2
end

@info "Done!"

# map(1:10) do x
#     2x
# end