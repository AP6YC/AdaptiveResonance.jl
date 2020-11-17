"""
    performance(y_hat, y)

Returns the categorization performance of y_hat against y.
"""
function performance(y_hat::Array, y::Array)
    # Compute the confusion matrix and calculate performance as trace/sum
    conf = confusion_matrix(categorical(y_hat), categorical(y), warn=false)
    return tr(conf.mat)/sum(conf.mat)
end
