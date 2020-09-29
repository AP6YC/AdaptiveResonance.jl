# module funcs

# export similarity, similarity_meta


"""
    similarity_meta(method, F2, field_name, gamma_ref)

Compute the similarity metric depending on method using meta programming to
access the correct field.
"""
function similarity_meta(method::String, F2, field_name::String, gamma_ref::AbstractFloat)
    @debug "Computing similarity"

    if field_name != "T" && field_name != "M"
        error("Incorrect field name for similarity metric.")
    end

    field = get_field_native(F2, field_name)

    # Single linkage
    if method == "single"
        value = maximum(field)
    # Average linkage
    elseif method == "average"
        value = mean(field)
    # Complete linkage
    elseif method == "complete"
        value = minimum(field)
    # Median linkage
    elseif method == "median"
        value = median(field)
    elseif method == "weighted"
        value = field' * (F2.n / sum(F2.n))
    elseif method == "centroid"
        Wc = minimum(F2.W)
        T = norm(min(sample, Wc), 1)
        if field_name == "T"
            value = T
        elseif field_name == "M"
            value = (norm(Wc, 1)^gamma_ref)*T
        end
    else
        error("Invalid/unimplemented similarity method")
    end
    return value
end # similarity

"""
    similarity(method, F2, field_name, gamma_ref)

Compute the similarity metric depending on method with explicit comparisons
for the field name.
"""
function similarity(method::String, F2, field_name::String, sample, gamma_ref::AbstractFloat)
    @debug "Computing similarity"

    if field_name != "T" && field_name != "M"
        error("Incorrect field name for similarity metric.")
    end
    # Single linkage
    if method == "single"
        if field_name == "T"
            value = maximum(F2.T)
        elseif field_name == "M"
            value = maximum(F2.M)
        end
    # Average linkage
    elseif method == "average"
        if field_name == "T"
            value = mean(F2.T)
        elseif field_name == "M"
            value = mean(F2.M)
        end
    # Complete linkage
    elseif method == "complete"
        if field_name == "T"
            value = minimum(F2.T)
        elseif field_name == "M"
            value = minimum(F2.M)
        end
    # Median linkage
    elseif method == "median"
        if field_name == "T"
            value = median(F2.T)
        elseif field_name == "M"
            value = median(F2.M)
        end
    # Weighted linkage
    elseif method == "weighted"
        if field_name == "T"
            value = F2.T * (F2.n / sum(F2.n))
        elseif field_name == "M"
            value = F2.M * (F2.n / sum(F2.n))
        end
    # Centroid linkage
    elseif method == "centroid"
        Wc = minimum(F2.W)
        T = norm(min(sample, Wc), 1)
        if field_name == "T"
            value = T
        elseif field_name == "M"
            value = (norm(Wc, 1)^gamma_ref)*T
        end
    else
        error("Invalid/unimplemented similarity method")
    end
end # similarity

# end