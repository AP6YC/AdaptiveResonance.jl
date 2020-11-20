mutable struct CONN <: AbstractCVI
    # dim::Array              # Dimension fo the input data
    n_samples::Int          # Number of samples encountered
    label_protos::Array     # Prototypes of each cluster
    CADJ::Array             # CADJ matrix
    CONN::Array             # CONN matrix
    n_clusters::Int         # Number of clusters
    n_prototypes::Int       # Number of prototypes
    inter_conn::Int         # Inter connectivity
    intra_conn::Int         # Intra connectivity
    criterion_value::Array  # Calculated conn_index
    inter_k_cache::Array    # Cached average inter conn for each cluster
    inter_kl_cache::Array   # Cached inter conn between each cluster
    intra_k_cache::Array    # Cached intra conn for each cluster
    condition::String       # Experimental switch for interconectivity (CONN or CADJ)
    missing_samples::Int    # Track samples with only a single bmu
end

function CONN()
    CONN(0,                             # n_samples
         Array{Float64}(undef, 0),      # label_protos
         Array{Float64}(undef, 0, 0),   # CADJ
         Array{Float64}(undef, 0, 0),   # CONN
         0,                             # n_clusters
         0,                             # n_prototypes
         0,                             # inter_conn
         0,                             # intra_conn
         Array{Float64}(undef, 0),      # criterion_value
         Array{Float64}(undef, 0),      # inter_k_cache
         Array{Float64}(undef, 0),      # inter_kl_cache
         Array{Float64}(undef, 0),      # intra_k_cache
         "CONN",                        # condition
         0                              # missing_samples
         )
end

function param_inc(cvi::CONN, p::Int, p2::Int, label::Int)
    cvi.n_samples += 1
    new_proto_flag = false
    if max(p, p2) > cvi.n_prototypes
        new_proto_flag = true
        cvi.n_prototypes += 1
        if p2 > 0
            cvi.CADJ[p, p2] = 1
            cvi.CADJ[p2, p] = 0
        else
            cvi.CADJ[p, 1] = 0
            cvi.CADJ[1, p] = 0
            cvi.missing_samples += 1
        end

    else
        if !(p == p2) && p2 > 0
            cvi.CADJ[p, p2] = cvi.CADJ[p, p2] + 1
        end
    end

    if p2 > 0
        if cvi.missing_samples > 0
            cvi.CADJ[p2, p] = cvi.CADJ[p2, p] + cvi.missing_samples
            cvi.missing_samples = 0
        end
        cvi.CONN[p, p2] = cvi.CADJ[p, p2] = cvi.CADJ[p2, p]
        cvi.CONN[p2, p] = cvi.CONN[p, p2]
    end

    if label > cvi.n_clusters
        cvi.n_clusters += 1
        cvi.label_protos[label] = p
    elseif new_proto_flag
        cvi.label_protos[label] = [cvi.label_protos[label] p] # TODO
    end
end