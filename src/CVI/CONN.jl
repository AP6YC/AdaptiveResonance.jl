mutable struct CONN <: AbstractCVI
    dim::Int64
    n_samples::Int64
    label_protos::Array{Array{Int64, 1}, 1}
    CADJ::Array{Float64, 2}
    CONN::Array{Float64, 2}
    n_clusters::Int64
    n_prototypes::Int64
    inter_conn::Float64
    intra_conn::Float64
    criterion_value::Float64
    inter_k_cache::Array{Float64, 1}
    inter_kl_cache::Array{Float64, 2}
    intra_k_cache::Array{Float64, 1}
    condition::String
    missing_samples::Int64
end

function CONN()
    CONN(
        0,                                      # dim
        0,                                      # n_samples
        Array{Array{Int64, 1}, 1}(undef, 0),    # label_protos
        Array{Float64, 2}(undef, 0, 0),         # CADJ
        Array{Float64, 2}(undef, 0, 0),         # CONN
        0,                                      # n_clusters
        0,                                      # n_prototypes
        0.0,                                    # inter_conn
        0.0,                                    # intra_conn
        0.0,                                    # criterion_value
        Array{Float64, 1}(undef, 0),            # inter_k_cache
        Array{Float64, 2}(undef, 0, 0),         # inter_kl_cache
        Array{Float64, 1}(undef, 0),            # intra_k_cache
        "CONN",                                 # condition
        0                                       # missing_samples
    )
end

"""
    expand_array(arr::Array{Float64, 2} ; n::Int64 = 1)

Expand the dimension of a 2-D array by n in each dimension (default 1).
"""
function expand_array(arr::Array{Float64, 2} ; n::Int64 = 1)
    dim, _ = size(arr)
    new_arr = zeros(dim+n, dim+n)
    new_arr[1:end-n, 1:end-n] = arr
    return new_arr
end # expand_array(arr::Array{Float64, 2} ; n::Int64 = 1)

"""
    expand_array(arr::Array{Float64, 1} ; n::Int64 = 1)

Expand a 1-D array by n zeros (default is 1).
"""
function expand_array(arr::Array{Float64, 1} ; n::Int64 = 1)
    dim = length(arr)
    new_arr = zeros(dim)
    new_arr[1:end-n] = arr
    return new_arr
end # expand_array(arr::Array{Float64, 1} ; n::Int64 = 1)

"""
    expand_array!(arr::Array{Float64} ; n::Int64 = 1)

Expand the dimension of a 1-D or 2-D array in place by n in each dimension (default n).

Accepts only 1-D or 2-D float arrays.
Does nothing if n = 0.
Throws an error if n < 0.
"""
function expand_array!(arr::Array{Float64} ; n::Int64 = 1)
    if n==0
        # If n is zero, make no change
        return
    elseif n < 0
        # Shrinking an array is not defined, so throw an error
        error("Trying to expand an array to a smaller dimension.")
    else
        # Otherwise, expand the array by n dimensions, padded with zeros
        arr = expand_array(arr ; n=n)
    end
end # expand_array!(arr::Array{Float64, 2} ; n::Int64 = 1)

"""
    expand_array_to!(arr::Array{Float64}, new_dim::Int64 = 0)

Expand the array in place to new_dim dimensions.

Accepts 1-D or 2-D float arrays.
Does nothing if the new dim is smaller than the old one.
"""
function expand_array_to!(arr::Array{Float64}, new_dim::Int64 = 0)
    # Get the correct dimensionality and number of samples
    if ndims(arr) > 1
        old_dim, _ = size(data)
    else
        old_dim = length(data)
    end

    # Get the number to expand the dims by
    n = new_dim - old_dim
    if n < 1
        # Do nothing if the new dimension is smaller than the old one
        return
    else
        # Expand the array in place by n dims
        expand_array!(arr, n=n)
    end
end # expand_array_to!(arr::Array{Float64}, new_dim::Int64 = 0)

"""
    intra_k(cvi::CONN, Ck::Int64)
"""
function intra_k(cvi::CONN, Ck::Int64)
    protos_k = cvi.label_protos[Ck]
    # 1. Compute the numerator
    # All samples in which 1st and 2nd BMUs belong to Ck
    temp1 = cvi.CADJ[protos_k, protos_k]
    ic1 = sum(temp1)
    # 2. Compute the denominator
    # All samples in which 1st BMUs belong to Ck
    temp2 = cvi.CADJ[protos_k, :]
    ic2 = sum(temp2)

    if ic2 > 0
        ic = ic1/ic2
    else
        # Return 0 if invalid div
        ic = 0
    end
    return ic
end # intra_k(cvi::CONN, Ck::Int64)

"""
    calc_intra_conn!(cvi::CONN, Ck::Int64)

Calcluate the intra conn for a cluster Ck.
"""
function calc_intra_conn!(cvi::CONN, Ck::Int64=0)
    # If a cluster is given
    if Ck != 0
        # Make room for the cluster
        expand_array_to!(cvi.intra_k_cache, Ck)
        # Update the cached value
        cvi.intra_k_cache[Ck] = cvi.intra_k[Ck]
    end
    cvi.intra_conn = 0.0
    for k=1:cvi.n_clusters
        # If no cluster is specified
        if Ck == 0
            # Make room for the cached cluster values
            expand_array_to!(cvi.intra_k_cache, k)
            cvi.intra_k_cache[k] = intra_k(cvi, k)
        end
        # Sum cached values
        cvi.intra_conn += cvi.intra_k_cache[k]
    end
    # Take the average of the cached values
    cvi.intra_conn /= cvi.n_clusters
end # calc_intra_conn!(cvi::CONN, Ck::Int64)

"""
    inter_kl(cvi::CONN, Ck::Int64, Cl::Int64)

Inter connectivity between two clusters Ck and Cl
"""
function inter_kl(cvi::CONN, Ck::Int64, Cl::Int64)
    protos_k = cvi.label_protos[Ck]
    protos_l = cvi.label_protos[Cl]
    if isapprox(Cl, Ck)
        # Identify prototypes that belong to cluster borders
        # Using CONN(i,j) > 0 as the condition
        if cvi.condition == "CONN"
            # Don't even ask
            protos_pkl = protos_k(vec(any(cvi.CONN(protos_k, protos_l), dims=2)))
        # Using CADJ(i,j) > 0 as the condition
        elseif cvi.condition == "CADJ"
            protos_pkl = protos_k(vec(any(cvi.CADJ(protos_k, protos_l), dims=2)))
        else
            error("Incompatible CVI condition: must be either CONN or CADJ.")
        end
        if isempty(protos_pkl)
            ic = 0
        else
            # Compute numerator
            CONN_kl = cvi.CONN[protos_pkl, protos_l]
            conn_ck_cl = sum(CONN_kl)
            # Compute denominator
            CONN_pkl = cvi.CONN[protos_pkl, [protos_k; protos_l]]
            intersection = cvi.CONN[protos_pkl, protos_pkl]
            # Subtract what has been counted twice (CONN is symmetric)
            conn_pkl = sum(CONN_pkl) - sum(intersection)/2.0
            # Compute inter_conn
            if conn_pkl > 0
                # Return quotient or zero if invalid div
                ic = conn_ck_cl / conn_pkl
            else
                ic = 0
            end
        end
    else
        ic = 0
    end
    return ic
end # inter_kl(cvi::CONN, Ck::Int64, Cl::Int64)

"""
    calc_inter_k!(cvi::CONN, Ck::Int64=0, Cl::Int64=0)
"""
function calc_inter_k!(cvi::CONN, Ck::Int64=0, Cl::Int64=0)
    if Ck != 0 && Cl != 0
        if isapprox(Ck, Cl)
            cvi.inter_kl_cache[Ck, Cl] = cvi.inter_kl[Ck, Cl]
            cvi.inter_kl_cache[Cl, Ck] = cvi.inter_kl[Cl, Ck]
            cvi.inter_k_cache[Ck] = maximum(cvi.inter_kl_cache[Ck, :])
            cvi.inter_k_cache[Cl] = maximum(cvi.inter_kl_cache[Cl, :])
        else
            for Cj = 1:cvi.n_clusters
                cvi.inter_kl_cache[Ck, Cj] = inter_kl(cvi, Ck, Cl)
            end
            cvi.inter_k_cache[Ck] = maximum(cvi.inter_kl_cache[Ck, :])
        end
    end
end # calc_inter_k!(cvi::CONN, Ck::Int64=0, Cl::Int64=0)

"""
    calc_inter_conn!(cvi::CONN, Ck::Int64=0, Cl::Int64=0)
"""
function calc_inter_conn!(cvi::CONN, Ck::Int64=0, Cl::Int64=0)
    if Ck != 0 && Cl != 0
        calc_inter_k(cvi, Ck, Cl)
    else
        for k = 1:cvi.n_clusters
            calc_inter_k!(cvi, k)
        end
    end
    cvi.inter_conn = sum(cvi.inter_k_cache) / cvi.n_clusters
end # calc_inter_conn!(cvi::CONN, Ck::Int64=0, Cl::Int64=0)

"""
    param_inc!(cvi::CONN, p::Int64, p2::Int64, label::Int64)
"""
function param_inc!(cvi::CONN, p::Int64, p2::Int64, label::Int64)
    # Increment the sample count
    cvi.n_samples += 1
    # New prototype flag defaults to false
    new_proto_flag = false
    # If there is a new prototype
    if max(p, p2) > cvi.n_prototypes
        # New prototype flag is raised
        new_proto_flag = true
        # Increment the prototype count
        cvi.n_prototypes += 1
        # Expand the array to the new number of prototypes
        expand_array_to!(cvi.CADJ, cvi.n_prototypes)
        # If there is more than one prototype
        if p2 > 0
            cvi.CADJ[p, p2] = 1
            cvi.CADJ[p2, p] = 0
        else
            # Otherwise initialize the CADJ matrix
            cvi.CADJ[p, 1] = 0
            cvi.CADJ[1, p] = 0
            # CADJ cannot track samples, so log the missing sample
            cvi.missing_samples += 1
        end
    else
        # If more than one prototype
        if p != p2 && p2 > 0
            # Increment the CADJ counter
            cvi.CADJ[p, p2] += 1
        end
    end

    # If more than one prototype
    if p2 > 0
        # If we have missing samples in the CADJ matrix
        if cvi.missing_samples > 0
            # Count samples in the CADJ matrix
            cvi.CADJ[p2, p] += cvi.missing_samples
            # Resent the missing samples counter
            cvi.missing_samples = 0
        end
        # Expand the CONN array if necessary
        expand_array_to!(cvi.CONN, cvi.n_prototypes)
        # Update the CONN matrix
        cvi.CONN[p, p2] = cvi.CADJ[p, p2] + cvi.CADJ[p2, p]
        cvi.CONN[p2, p] = cvi.CONN[p, p2]
    end

    # If the label is to a wholly new cluster
    if label > cvi.n_clusters
        # Update cluster count and prototype list
        cvi.n_clusters += 1
        push!(cvi.label_protos, [p])
    # Otherwise, if there is a new prototype for this cluster
    elseif new_proto_flag
        # Remember this cluster-prototype pair
        push!(cvi.label_protos[label], p)
    end
    # Update the cluster intra_conn
    calc_intra_conn!(cvi, label)
    if p2 > 0
        for Cl = 1:cvi.n_clusters
            if any(x -> x > 0, cvi.label_protos[Cl])
                calc_inter_conn!(cvi, label, Cl)
            end
        end
    end
end # param_inc!(cvi::CONN, p::Int64, p2::Int64, label::Int64)

# function