@testset "CVIs" begin
    @info "CVI Testing"

    # Construct the cvis
    cvis = [
        XB(),
        DB(),
        PS()
    ]
    n_cvis = length(cvis)

    # Grab all the data paths for testing
    data_paths = readdir("../data/cvi", join=true)

    # Initialize the data containers
    data, labels, n_samples = Dict(), Dict(), Dict()

    # Sanitize all the data
    p = 0.3
    @info "Subsampling data at rate: $p"
    for data_path in data_paths
        # Load the data, get a subset, and relabel in order
        local_data, local_labels = get_cvi_data(data_path)
        local_data, local_labels = get_bernoulli_subset(local_data, local_labels, p)
        local_labels = relabel_cvi_data(local_labels)

        # Store the sanitized data
        data[data_path] = local_data
        labels[data_path] = local_labels
        n_samples[data_path] = length(local_labels)
    end

    # Incremental
    @info "------- CVI Incremental -------"
    cvi_i = Dict()
    for data_path in data_paths
        @info "Data: $data_path"
        cvi_i[data_path] = deepcopy(cvis)
        for cvi in cvi_i[data_path]
            # @info "ICVI: $(typeof(cvi))"
            for ix = 1:n_samples[data_path]
                sample = data[data_path][:, ix]
                label = labels[data_path][ix]
                # param_inc!(cvi, data[:, ix], labels[ix])
                param_inc!(cvi, sample, label)
                evaluate!(cvi)
            end
        end
    end

    # Batch
    @info "------- CVI Batch -------"
    cvi_b = Dict()
    for data_path in data_paths
        @info "Data: $data_path"
        cvi_b[data_path] = deepcopy(cvis)
        for cvi in cvi_b[data_path]
            # @info "CVI: $(typeof(cvi))"
            param_batch!(cvi, data[data_path], labels[data_path])
            evaluate!(cvi)
        end
    end

    # Incremental porcelain
    @info "------- ICVI Porcelain -------"
    cvi_ip = Dict()
    # cvs_ip = Dict()
    for data_path in data_paths
        @info "Data: $data_path"
        cvi_ip[data_path] = deepcopy(cvis)
        # cvs_ip[data_path] = zeros(n_samples[data_path], n_cvis)
        # for cx = 1:n_cvis
        for cvi in cvi_ip[data_path]
            for ix = 1:n_samples[data_path]
                sample = data[data_path][:, ix]
                label = labels[data_path][ix]
                # _ = get_icvi!(cvi_ip[data_path][cx], sample, label)
                _ = get_icvi!(cvi, sample, label)
                # cvs_ip[data_path][ix, cx] = cv
            end
        end
    end

    # Batch porcelain
    @info "------- CVI Porcelain -------"
    cvi_bp = Dict()
    for data_path in data_paths
        @info "Data: $data_path"
        cvi_bp[data_path] = deepcopy(cvis)
        # cvs_b = zeros(n_cvis)
        # for cx = 1:n_cvis
        for cvi in cvi_bp[data_path]
            # cvs_b[cx] = get_cvi!(cvi_bp[cx], data, labels)
            # _ = get_cvi!(cvi_bp[data_path][cx], data[data_path], labels[data_path])
            _ = get_cvi!(cvi, data[data_path], labels[data_path])
        end
    end

    # Test that all permutations are equivalent
    for data_path in data_paths
        for cx = 1:n_cvis
            # I to B
            @test isapprox(cvi_i[data_path][cx].criterion_value,
                cvi_b[data_path][cx].criterion_value)

            # IP to BP
            @test isapprox(cvi_ip[data_path][cx].criterion_value,
                cvi_bp[data_path][cx].criterion_value)

            # I to IP
            @test isapprox(cvi_i[data_path][cx].criterion_value,
                cvi_ip[data_path][cx].criterion_value)

            # B to BP
            @test isapprox(cvi_b[data_path][cx].criterion_value,
                cvi_bp[data_path][cx].criterion_value)
        end
    end
end
