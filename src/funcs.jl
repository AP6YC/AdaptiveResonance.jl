"""
    complement_code(data)

Normalize the data x to [0, 1] and returns the augmented vector [x, 1 - x].
"""
function complement_code(data::Array)
    # Complement code the data and return a concatenated matrix

    # Get the correct dimensionality and number of samples
    if ndims(data) > 1
        dim, n_samples = size(data)
    else
        dim = 1
        n_samples = length(data)
    end
    x_raw = zeros(dim, n_samples)

    mins = [minimum(data[i, :]) for i in 1:dim]
    maxs = [maximum(data[i, :]) for i in 1:dim]

    for i = 1:dim
        if maxs[i] - mins[i] != 0
            x_raw[i, :] = (data[i, :] .- mins[i]) ./ (maxs[i] - mins[i])
        end
    end

    x = vcat(x_raw, 1 .- x_raw)
    return x
end

"""
    element_min(x::Array, W::Array)

Returns the element-wise minimum between sample x and weight W.
"""
function element_min(x::Array, W::Array)
    # Compute the element-wise minimum of two vectors
    return minimum([x W], dims = 2)
end

"""
    performance(y_hat, y)

Returns the categorization performance of y_hat against y.
"""
function performance(y_hat::Array, y::Array)
    # Compute the confusion matrix and calculate performance as trace/sum
    conf = confusion_matrix(categorical(y_hat), categorical(y), warn=false)
    return tr(conf.mat)/sum(conf.mat)
end

# """
#     get_field_meta(obj, field_name)

# Get the value of a struct's field using meta programming.
# """
# function get_field_meta(obj::Any, field_name::String)
#     field = Symbol(field_name)
#     code = quote
#         (obj) -> obj.$field
#     end
#     return eval(code)
# end

# """
#     get_field_native(obj, field_name)

# Get the value of a struct's field through the julia native method.
# """
# function get_field_native(obj::Any, field_name::String)
#     return getfield(obj, Symbol(field_name))
# end

# """
#     similarity_meta(method, F2, field_name, gamma_ref)

# Compute the similarity metric depending on method using meta programming to
# access the correct field.
# """
# function similarity_meta(method::String, F2, field_name::String, gamma_ref::AbstractFloat)
#     @debug "Computing similarity"

#     if field_name != "T" && field_name != "M"
#         error("Incorrect field name for similarity metric.")
#     end

#     field = get_field_native(F2, field_name)

#     # Single linkage
#     if method == "single"
#         value = maximum(field)
#     # Average linkage
#     elseif method == "average"
#         value = mean(field)
#     # Complete linkage
#     elseif method == "complete"
#         value = minimum(field)
#     # Median linkage
#     elseif method == "median"
#         value = median(field)
#     elseif method == "weighted"
#         value = field' * (F2.n / sum(F2.n))
#     elseif method == "centroid"
#         Wc = minimum(F2.W)
#         T = norm(min(sample, Wc), 1)
#         if field_name == "T"
#             value = T
#         elseif field_name == "M"
#             value = (norm(Wc, 1)^gamma_ref)*T
#         end
#     else
#         error("Invalid/unimplemented similarity method")
#     end
#     return value
# end # similarity

# """
#     similarity(method, F2, field_name, gamma_ref)

# Compute the similarity metric depending on method with explicit comparisons
# for the field name.
# """
# function similarity(method::String, F2, field_name::String, sample, gamma_ref::AbstractFloat)
#     @debug "Computing similarity"

#     if field_name != "T" && field_name != "M"
#         error("Incorrect field name for similarity metric.")
#     end
#     # Single linkage
#     if method == "single"
#         if field_name == "T"
#             value = maximum(F2.T)
#         elseif field_name == "M"
#             value = maximum(F2.M)
#         end
#     # Average linkage
#     elseif method == "average"
#         if field_name == "T"
#             value = mean(F2.T)
#         elseif field_name == "M"
#             value = mean(F2.M)
#         end
#     # Complete linkage
#     elseif method == "complete"
#         if field_name == "T"
#             value = minimum(F2.T)
#         elseif field_name == "M"
#             value = minimum(F2.M)
#         end
#     # Median linkage
#     elseif method == "median"
#         if field_name == "T"
#             value = median(F2.T)
#         elseif field_name == "M"
#             value = median(F2.M)
#         end
#     # Weighted linkage
#     elseif method == "weighted"
#         if field_name == "T"
#             value = F2.T * (F2.n / sum(F2.n))
#         elseif field_name == "M"
#             value = F2.M * (F2.n / sum(F2.n))
#         end
#     # Centroid linkage
#     elseif method == "centroid"
#         Wc = minimum(F2.W)
#         T = norm(min(sample, Wc), 1)
#         if field_name == "T"
#             value = T
#         elseif field_name == "M"
#             value = (norm(Wc, 1)^gamma_ref)*T
#         end
#     else
#         error("Invalid/unimplemented similarity method")
#     end
# end # similarity
