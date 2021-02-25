mutable struct CONN <: AbstractCVI
    dim::Int64
    n_samples::Int64
    label_protos::Array{Array{Int64, 1}, 1}
    CADJ::Array{Float64, 2}
    CONN::Array{Float64, 2}
    n_clusters::Int64
    n_prototypes::Int64
    inter_conn::Int64
    intra_conn::Int64
    criterion_values::Float64
    inter_k_cache::Array{Float64, 2}
    inter_kl_cache::Array{Float64, 2}
    intra_k_cache::Array{Float64, 2}
    condition::String
    missing_samples::Int64
end

function CONN()
    CONN(
        0,                              # dim
        0,                              # n_samples
        Array{Array{Int64, 1}, 1}(undef, 0), # label_protos
        Array{Float64, 2}(undef, 0, 0), # CADJ
        Array{Float64, 2}(undef, 0, 0), # CONN
        0,                              # n_clusters
        0,                              # n_prototypes
        0,                              # inter_conn
        0,                              # intra_conn
        0.0,                            # criterion_values
        Array{Float64, 2}(undef, 0, 0), # inter_k_cache
        Array{Float64, 2}(undef, 0, 0), # inter_kl_cache
        Array{Float64, 2}(undef, 0, 0), # intra_k_cache
        "CONN",                         # condition
        0                               # missing_samples
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
    expand_array!(arr::Array{Float64, 2} ; n::Int64 = 1)

Expand the dimension of a 2-D array in place by n in each dimension (default n).
"""
function expand_array!(arr::Array{Float64, 2} ; n::Int64 = 1)
    arr = expand_array(arr ; n=n)
end # expand_array!(arr::Array{Float64, 2} ; n::Int64 = 1)

"""
    calc_intra_conn(cvi::CONN, Ck::Int64)
"""
function calc_intra_conn(cvi::CONN ; Ck::Int64=0)
    if Ck != 0
        cvi.intra_k_cache[Ck]
    end
end # calc_intra_conn(cvi::CONN, Ck::Int64)

"""
    param_inc(cvi::CONN, p::Int64, p2::Int64, label::Int64)
"""
function param_inc(cvi::CONN, p::Int64, p2::Int64, label::Int64)
    cvi.n_samples += 1
    new_proto_flag = false
    if max(p, p2) > cvi.n_prototypes
        new_proto_flag = true
        cvi.n_prototypes += 1
        expand_array!(cvi.CADJ)
        if p2 > 0
            cvi.CADJ[p, p2] = 1
            cvi.CADJ[p2, p] = 0
        else
            cvi.CADJ[p, 1] = 0
            cvi.CADJ[1, p] = 0
            cvi.missing_samples += 1
        end
    else
        if p != p2 && p2 > 0
            cvi.CADJ[p, p2] += 1
        end
    end

    if p2 > 0
        if cvi.missing_samples > 0
            cvi.CADJ[p2, p] += cvi.missing_samples
            cvi.missing_samples = 0
        end
        conn_size = size(cvi.CONN)[1]
        if conn_size > cvi.n_prototypes
            expand_array!(cvi.CONN, n=conn_size-cvi.n_prototypes)
        end
        cvi.CONN[p, p2] = cvi.CADJ[p, p2] + cvi.CADJ[p2, p]
        cvi.CONN[p2, p] = cvi.CONN[p, p2]
    end

    if label > cvi.n_clusters
        cvi.n_clusters += 1
        push!(cvi.label_protos, [p])
    elseif new_proto_flag
        push!(cvi.label_protos[label], p)
    end
    cvi
end
