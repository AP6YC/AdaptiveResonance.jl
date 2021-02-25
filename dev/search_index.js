var documenterSearchIndex = {"docs":
[{"location":"man/contributing/#Contributing","page":"Contributing","title":"Contributing","text":"","category":"section"},{"location":"man/contributing/","page":"Contributing","title":"Contributing","text":"To contribute to the package, please follow the usual branch pull request procedure:","category":"page"},{"location":"man/contributing/","page":"Contributing","title":"Contributing","text":"Fork the project.\nCreate your feature branch (git checkout -b my-new-feature).\nCommit your changes (git commit -am 'Added some feature').\nPush to the branch (git push origin my-new-feature).\nCreate a new GitHub pull request.","category":"page"},{"location":"man/full-index/#main-index","page":"Index","title":"Index","text":"","category":"section"},{"location":"man/full-index/","page":"Index","title":"Index","text":"Pages = [\"lib/public.md\"]","category":"page"},{"location":"man/full-index/","page":"Index","title":"Index","text":"Modules = [AdaptiveResonance]","category":"page"},{"location":"man/full-index/#AdaptiveResonance.DAM","page":"Index","title":"AdaptiveResonance.DAM","text":"DAM <: AbstractART\n\nDefault ARTMAP struct.\n\n\n\n\n\n","category":"type"},{"location":"man/full-index/#AdaptiveResonance.DAM-Tuple{opts_DAM}","page":"Index","title":"AdaptiveResonance.DAM","text":"DAM(opts)\n\nImplements a Default ARTMAP learner with specified options\n\nExamples\n\njulia> opts = opts_DAM()\njulia> DAM(opts)\nDAM\n    opts: opts_DAM\n    ...\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.DAM-Tuple{}","page":"Index","title":"AdaptiveResonance.DAM","text":"DAM()\n\nImplements a Default ARTMAP learner.\n\nExamples\n\njulia> DAM()\nDAM\n    opts: opts_DAM\n    ...\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.DDVFA","page":"Index","title":"AdaptiveResonance.DDVFA","text":"DDVFA <: AbstractART\n\nDistributed Dual Vigilance Fuzzy ARTMAP module struct.\n\nExamples\n\njulia> DDVFA()\nDDVFA\n    opts: opts_DDVFA\n    supopts::opts_GNFA\n    ...\n\n\n\n\n\n","category":"type"},{"location":"man/full-index/#AdaptiveResonance.DDVFA-Tuple{opts_DDVFA}","page":"Index","title":"AdaptiveResonance.DDVFA","text":"DDVFA(opts::opts_DDVFA)\n\nImplements a DDVFA learner with specified options.\n\nExamples\n\njulia> my_opts = opts_DDVFA()\njulia> DDVFA(my_opts)\nDDVFA\n    opts: opts_DDVFA\n    supopts: opts_GNFA\n    ...\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.DDVFA-Tuple{}","page":"Index","title":"AdaptiveResonance.DDVFA","text":"DDVFA()\n\nImplements a DDVFA learner with default options.\n\nExamples\n\njulia> DDVFA()\nDDVFA\n    opts: opts_DDVFA\n    supopts: opts_GNFA\n    ...\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.DataConfig","page":"Index","title":"AdaptiveResonance.DataConfig","text":"DataConfig\n\nConatiner to standardize training/testing data configuration.\n\n\n\n\n\n","category":"type"},{"location":"man/full-index/#AdaptiveResonance.DataConfig-Tuple{Array,Array}","page":"Index","title":"AdaptiveResonance.DataConfig","text":"DataConfig(mins::Array, maxs::Array)\n\nConvenience constructor for DataConfig, requiring only mins and maxs of the features.\n\nThis constructor is used when the mins and maxs differ across features. The dimension is inferred by the length of the mins and maxs.\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.DataConfig-Tuple{Real,Real,Int64}","page":"Index","title":"AdaptiveResonance.DataConfig","text":"DataConfig(min::Real, max::Real, dim::Int64)\n\nConvenience constructor for DataConfig, requiring only a global min, max, and dim.\n\nThis constructor is used in the case that the feature mins and maxs are all the same respectively.\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.DataConfig-Tuple{}","page":"Index","title":"AdaptiveResonance.DataConfig","text":"DataConfig()\n\nDefault constructor for a data configuration, not set up.\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.FAM","page":"Index","title":"AdaptiveResonance.FAM","text":"FAM\n\nFuzzy ARTMAP struct.\n\n\n\n\n\n","category":"type"},{"location":"man/full-index/#AdaptiveResonance.FAM-Tuple{opts_FAM}","page":"Index","title":"AdaptiveResonance.FAM","text":"FAM(opts)\n\nImplements a Fuzzy ARTMAP learner with specified options.\n\nExamples\n\njulia> opts = opts_FAM()\njulia> FAM(opts)\nFAM\n    opts: opts_FAM\n    ...\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.FAM-Tuple{}","page":"Index","title":"AdaptiveResonance.FAM","text":"FAM()\n\nImplements a Fuzzy ARTMAP learner.\n\nExamples\n\njulia> FAM()\nFAM\n    opts: opts_FAM\n    ...\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.GNFA","page":"Index","title":"AdaptiveResonance.GNFA","text":"GNFA <: AbstractART\n\nGamma-Normalized Fuzzy ART learner struct\n\nExamples\n\njulia> GNFA()\nGNFA\n    opts: opts_GNFA\n    ...\n\n\n\n\n\n","category":"type"},{"location":"man/full-index/#AdaptiveResonance.GNFA-Tuple{opts_GNFA,Array}","page":"Index","title":"AdaptiveResonance.GNFA","text":"GNFA(opts::opts_GNFA, sample::Array)\n\nCreate and initialize a GNFA with a single sample in one step.\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.GNFA-Tuple{opts_GNFA}","page":"Index","title":"AdaptiveResonance.GNFA","text":"GNFA(opts::opts_GNFA)\n\nImplements a Gamma-Normalized Fuzzy ART learner with specified options.\n\nExamples\n\njulia> GNFA(opts)\nGNFA\n    opts: opts_GNFA\n    ...\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.GNFA-Tuple{}","page":"Index","title":"AdaptiveResonance.GNFA","text":"GNFA()\n\nImplements a Gamma-Normalized Fuzzy ART learner.\n\nExamples\n\njulia> GNFA()\nGNFA\n    opts: opts_GNFA\n    ...\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.SFAM","page":"Index","title":"AdaptiveResonance.SFAM","text":"SFAM\n\nSimple Fuzzy ARTMAP struct.\n\n\n\n\n\n","category":"type"},{"location":"man/full-index/#AdaptiveResonance.SFAM-Tuple{opts_SFAM}","page":"Index","title":"AdaptiveResonance.SFAM","text":"SFAM(opts)\n\nImplements a Simple Fuzzy ARTMAP learner with specified options.\n\nExamples\n\njulia> opts = opts_SFAM()\njulia> SFAM(opts)\nSFAM\n    opts: opts_SFAM\n    ...\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.SFAM-Tuple{}","page":"Index","title":"AdaptiveResonance.SFAM","text":"SFAM()\n\nImplements a Simple Fuzzy ARTMAP learner.\n\nExamples\n\njulia> SFAM()\nSFAM\n    opts: opts_SFAM\n    ...\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.XB","page":"Index","title":"AdaptiveResonance.XB","text":"XB\n\nThe stateful information of the Xie-Beni CVI.\n\n\n\n\n\n","category":"type"},{"location":"man/full-index/#AdaptiveResonance.XB-Tuple{}","page":"Index","title":"AdaptiveResonance.XB","text":"XB()\n\nDefault constructor for the Xie-Beni (XB) Cluster Validity Index.\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.opts_DAM","page":"Index","title":"AdaptiveResonance.opts_DAM","text":"opts_DAM()\n\nImplements a Default ARTMAP learner's options.\n\nExamples\n\njulia> my_opts = opts_DAM()\n\n\n\n\n\n","category":"type"},{"location":"man/full-index/#AdaptiveResonance.opts_DDVFA","page":"Index","title":"AdaptiveResonance.opts_DDVFA","text":"opts_DDVFA()\n\nDistributed Dual Vigilance Fuzzy ART options struct.\n\nExamples\n\njulia> opts_DDVFA()\nInitialized opts_DDVFA\n\n\n\n\n\n","category":"type"},{"location":"man/full-index/#AdaptiveResonance.opts_FAM","page":"Index","title":"AdaptiveResonance.opts_FAM","text":"opts_FAM()\n\nImplements a Fuzzy ARTMAP learner's options.\n\nExamples\n\njulia> my_opts = opts_FAM()\n\n\n\n\n\n","category":"type"},{"location":"man/full-index/#AdaptiveResonance.opts_GNFA","page":"Index","title":"AdaptiveResonance.opts_GNFA","text":"opts_GNFA()\n\nGamma-Normalized Fuzzy ART options struct.\n\nExamples\n\njulia> opts_GNFA()\nInitialized GNFA\n\n\n\n\n\n","category":"type"},{"location":"man/full-index/#AdaptiveResonance.opts_SFAM","page":"Index","title":"AdaptiveResonance.opts_SFAM","text":"opts_SFAM()\n\nImplements a Simple Fuzzy ARTMAP learner's options.\n\nExamples\n\njulia> my_opts = opts_SFAM()\n\n\n\n\n\n","category":"type"},{"location":"man/full-index/#AdaptiveResonance.activation-Tuple{DAM,Array,Array}","page":"Index","title":"AdaptiveResonance.activation","text":"activation(art::DAM, x::Array, W::Array)\n\nDefault ARTMAP's choice-by-difference activation function.\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.activation-Tuple{SFAM,Array,Array}","page":"Index","title":"AdaptiveResonance.activation","text":"activation(art::SFAM, x::Array, W::Array)\n\nReturns the activation value of the Simple Fuzzy ARTMAP module with weight W and sample x.\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.activation_match!-Tuple{GNFA,Array}","page":"Index","title":"AdaptiveResonance.activation_match!","text":"activation_match!(art::GNFA, x::Array)\n\nComputes the activation and match functions of the art module against sample x.\n\nExamples\n\njulia> my_GNFA = GNFA()\nGNFA\n    opts: opts_GNFA\n    ...\njulia> x, y = load_data()\njulia> train!(my_GNFA, x)\njulia> x_sample = x[:, 1]\njulia> activation_match!(my_GNFA, x_sample)\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.art_match-Tuple{DAM,Array,Array}","page":"Index","title":"AdaptiveResonance.art_match","text":"art_match(art::DAM, x::Array, W::Array)\n\nReturns the match function for the Default ARTMAP module with weight W and sample x.\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.art_match-Tuple{SFAM,Array,Array}","page":"Index","title":"AdaptiveResonance.art_match","text":"art_match(art::SFAM, x::Array, W::Array)\n\nReturns the match function for the Simple Fuzzy ARTMAP module with weight W and sample x.\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.artscene_filter-Union{Tuple{Array{T,3}}, Tuple{T}} where T<:Real","page":"Index","title":"AdaptiveResonance.artscene_filter","text":"artscene_filter(raw_image::Array{T, 3} ;  distributed::Bool=true) where {T<:Real}\n\nProcess the full artscene filter toolchain on an image.\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.calc_inter_conn!","page":"Index","title":"AdaptiveResonance.calc_inter_conn!","text":"calc_inter_conn!(cvi::CONN, Ck::Int64=0, Cl::Int64=0)\n\n\n\n\n\n","category":"function"},{"location":"man/full-index/#AdaptiveResonance.calc_inter_k!","page":"Index","title":"AdaptiveResonance.calc_inter_k!","text":"calc_inter_k!(cvi::CONN, Ck::Int64=0, Cl::Int64=0)\n\n\n\n\n\n","category":"function"},{"location":"man/full-index/#AdaptiveResonance.calc_intra_conn!","page":"Index","title":"AdaptiveResonance.calc_intra_conn!","text":"calc_intra_conn!(cvi::CONN, Ck::Int64)\n\nCalcluate the intra conn for a cluster Ck.\n\n\n\n\n\n","category":"function"},{"location":"man/full-index/#AdaptiveResonance.classify-Tuple{DAM,Array}","page":"Index","title":"AdaptiveResonance.classify","text":"classify(art::DAM, x::Array ; preprocessed=false)\n\nCategorize data 'x' using a trained Default ARTMAP module 'art'.\n\nExamples\n\njulia> x, y = load_data()\njulia> x_test, y_test = load_test_data()\njulia> art = DAM()\nDAM\n    opts: opts_DAM\n    ...\njulia> train!(art, x, y)\njulia> classify(art, x_test)\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.classify-Tuple{DDVFA,Array}","page":"Index","title":"AdaptiveResonance.classify","text":"classify(art::DDVFA, x::Array ; preprocessed=false)\n\nPredict categories of 'x' using the DDVFA model.\n\nReturns predicted categories 'y_hat.'\n\nExamples\n\njulia> my_DDVFA = DDVFA()\nDDVFA\n    opts: opts_DDVFA\n    ...\njulia> x, y = load_data()\njulia> train!(my_DDVFA, x)\njulia> y_hat = classify(my_DDVFA, y)\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.classify-Tuple{GNFA,Array}","page":"Index","title":"AdaptiveResonance.classify","text":"classify(art::GNFA, x::Array)\n\nPredict categories of 'x' using the GNFA model.\n\nReturns predicted categories 'y_hat'\n\nExamples\n\njulia> my_GNFA = GNFA()\nGNFA\n    opts: opts_GNFA\n    ...\njulia> x, y = load_data()\njulia> train!(my_GNFA, x)\njulia> y_hat = classify(my_GNFA, y)\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.classify-Tuple{SFAM,Array}","page":"Index","title":"AdaptiveResonance.classify","text":"classify(art::SFAM, x::Array ; preprocessed=false)\n\nCategorize data 'x' using a trained Simple Fuzzy ARTMAP module 'art'.\n\nExamples\n\njulia> x, y = load_data()\njulia> x_test, y_test = load_test_data()\njulia> art = SFAM()\nSFAM\n    opts: opts_SFAM\n    ...\njulia> train!(art, x, y)\njulia> classify(art, x_test)\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.color_to_gray-Tuple{Array}","page":"Index","title":"AdaptiveResonance.color_to_gray","text":"color_to_gray(image::Array)\n\nARTSCENE Stage 1: Color-to-gray image transformation.\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.competition_kernel-Tuple{Int64,Int64}","page":"Index","title":"AdaptiveResonance.competition_kernel","text":"competition_kernel(l::Int, k::Int ; sign::String=\"plus\")\n\nCompetition kernel for ARTSCENE: Stage 5.\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.complement_code-Tuple{Array}","page":"Index","title":"AdaptiveResonance.complement_code","text":"complement_code(data::Array ; config::DataConfig=DataConfig())\n\nNormalize the data x to [0, 1] and returns the augmented vector [x, 1 - x].\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.contrast_insensitive_oriented_filtering-Tuple{Array}","page":"Index","title":"AdaptiveResonance.contrast_insensitive_oriented_filtering","text":"contrast_insensitive_oriented_filtering(y::Array)\n\nARTSCENE Stage 4: Contrast-insensitive oriented filtering.\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.contrast_normalization-Tuple{Array}","page":"Index","title":"AdaptiveResonance.contrast_normalization","text":"contrast_normalization(image::Array ; distributed::Bool=true)\n\nARTSCENE Stage 2: Constrast normalization.\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.contrast_sensitive_oriented_filtering-Tuple{Array,Array}","page":"Index","title":"AdaptiveResonance.contrast_sensitive_oriented_filtering","text":"contrast_sensitive_oriented_filtering(image::Array, x::Array ; distributed::Bool=true)\n\nARTSCENE Stage 3: Contrast-sensitive oriented filtering.\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.data_setup!-Tuple{DataConfig,Array}","page":"Index","title":"AdaptiveResonance.data_setup!","text":"data_setup!(config::DataConfig, data::Array)\n\nSets up the data config for the ART module before training.\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.ddt_x-Tuple{Array,Array,Array,Bool}","page":"Index","title":"AdaptiveResonance.ddt_x","text":"ddt_x(x::Array, image::Array, sigma_s::Array, distributed::Bool)\n\nTime rate of change of LGN network (ARTSCENE Stage 2).\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.ddt_y-Tuple{Array,Array,Array,Real,Bool}","page":"Index","title":"AdaptiveResonance.ddt_y","text":"ddt_y(y::Array, X_plus::Array, X_minus::Array, alpha::Real, distributed::Bool)\n\nShunting equation for ARTSCENE Stage 3.\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.ddt_z-Tuple{Array}","page":"Index","title":"AdaptiveResonance.ddt_z","text":"ddt_z(z::Array ; distributed=true)\n\nTime rate of change for ARTSCENE: Stage 5.\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.element_min-Tuple{Array,Array}","page":"Index","title":"AdaptiveResonance.element_min","text":"element_min(x::Array, W::Array)\n\nReturns the element-wise minimum between sample x and weight W.\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.expand_array!","page":"Index","title":"AdaptiveResonance.expand_array!","text":"expand_array!(arr::Array{Float64}, new_dim::Int64 = 0)\n\nExpand the array in place to new_dim dimensions.\n\nAccepts 1-D or 2-D float arrays. Does nothing if the new dim is smaller than the old one.\n\n\n\n\n\n","category":"function"},{"location":"man/full-index/#AdaptiveResonance.expand_array!-Tuple{Array{Float64,N} where N}","page":"Index","title":"AdaptiveResonance.expand_array!","text":"expand_array!(arr::Array{Float64} ; n::Int64 = 1)\n\nExpand the dimension of a 1-D or 2-D array in place by n in each dimension (default n).\n\nAccepts only 1-D or 2-D float arrays. Does nothing if n = 0. Throws an error if n < 0.\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.expand_array-Tuple{Array{Float64,1}}","page":"Index","title":"AdaptiveResonance.expand_array","text":"expand_array(arr::Array{Float64, 1} ; n::Int64 = 1)\n\nExpand a 1-D array by n zeros (default is 1).\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.expand_array-Tuple{Array{Float64,2}}","page":"Index","title":"AdaptiveResonance.expand_array","text":"expand_array(arr::Array{Float64, 2} ; n::Int64 = 1)\n\nExpand the dimension of a 2-D array by n in each dimension (default 1).\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.get_data_characteristics-Tuple{Array}","page":"Index","title":"AdaptiveResonance.get_data_characteristics","text":"get_data_characteristics(data::Array ; config::DataConfig=DataConfig())\n\nGet the characteristics of the data, taking account if a data config is passed.\n\nIf no DataConfig is passed, then the data characteristics come from the array itself. Otherwise, use the config for the statistics of the data and the data array for the number of samples.\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.get_data_shape-Tuple{Array}","page":"Index","title":"AdaptiveResonance.get_data_shape","text":"get_data_shape(data::Array)\n\nReturns the correct feature dimension and number of samples.\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.get_icvi-Union{Tuple{M}, Tuple{N}, Tuple{T}, Tuple{T,Array{N,1},M}} where M<:Int64 where N<:Real where T<:AbstractCVI","page":"Index","title":"AdaptiveResonance.get_icvi","text":"get_icvi(cvi::T, x::Array{N, 1}, y::M) where {T<:AbstractCVI, N<:Real, M<:Int}\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.get_n_samples-Tuple{Array}","page":"Index","title":"AdaptiveResonance.get_n_samples","text":"get_n_samples(data::Array)\n\nReturns the number of samples, accounting for 1-D and 2-D arrays.\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.initialize!-Tuple{GNFA,Array}","page":"Index","title":"AdaptiveResonance.initialize!","text":"initialize!(art::GNFA, x::Array)\n\nInitializes a GNFA learner with an intial sample 'x'.\n\nExamples\n\njulia> my_GNFA = GNFA()\nGNFA\n    opts: opts_GNFA\n    ...\njulia> initialize!(my_GNFA, [1 2 3 4])\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.inter_kl-Tuple{CONN,Int64,Int64}","page":"Index","title":"AdaptiveResonance.inter_kl","text":"inter_kl(cvi::CONN, Ck::Int64, Cl::Int64)\n\nInter connectivity between two clusters Ck and Cl\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.intra_k-Tuple{CONN,Int64}","page":"Index","title":"AdaptiveResonance.intra_k","text":"intra_k(cvi::CONN, Ck::Int64)\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.learn!-Tuple{GNFA,Array,Int64}","page":"Index","title":"AdaptiveResonance.learn!","text":"learn!(art::GNFA, x::Array, index::Int)\n\nIn place learning function with instance counting.\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.learn-Tuple{DAM,Array,Array}","page":"Index","title":"AdaptiveResonance.learn","text":"learn(art::DAM, x::Array, W::Array)\n\nReturns a single updated weight for the Simple Fuzzy ARTMAP module for weight vector W and sample x.\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.learn-Tuple{GNFA,Array,Array}","page":"Index","title":"AdaptiveResonance.learn","text":"learn(art::GNFA, x::Array, W::Array)\n\nReturn the modified weight of the art module conditioned by sample x.\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.learn-Tuple{SFAM,Array,Array}","page":"Index","title":"AdaptiveResonance.learn","text":"learn(art::SFAM, x::Array, W::Array)\n\nReturns a single updated weight for the Simple Fuzzy ARTMAP module for weight vector W and sample x.\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.linear_normalization-Tuple{Array}","page":"Index","title":"AdaptiveResonance.linear_normalization","text":"linear_normalization(data::Array ; config::DataConfig=DataConfig())\n\nNormalize the data to the range [0, 1] along each feature.\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.orientation_competition-Tuple{Array}","page":"Index","title":"AdaptiveResonance.orientation_competition","text":"orientation_competition(z::Array)\n\nARTSCENE Stage 5: Orientation competition at the same position.\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.oriented_kernel-Tuple{Int64,Int64,Int64,Int64,Int64,Real,Real}","page":"Index","title":"AdaptiveResonance.oriented_kernel","text":"oriented_kernel(i::Int, j::Int, p::Int, q::Int, k::Int, sigma_h::Real, sigma_v::Real ; sign::String=\"plus\")\n\nOriented, elongated, spatially offset kernel G for ARTSCENE Stage 3.\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.param_batch!-Union{Tuple{I}, Tuple{T}, Tuple{XB,Array{T,2},Array{I,1}}} where I<:Int64 where T<:Real","page":"Index","title":"AdaptiveResonance.param_batch!","text":"param_batch!(cvi::XB, data::Array{T, 2}, labels::Array{I, 1}) where {T<:Real, I<:Int}\n\nCompute the XB CVI in batch.\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.param_inc!-Union{Tuple{I}, Tuple{T}, Tuple{XB,Array{T,1},I}} where I<:Int64 where T<:Real","page":"Index","title":"AdaptiveResonance.param_inc!","text":"param_inc!(cvi::XB, sample::Array{T, 1}, label::I) where {T<:Real, I<:Int}\n\nCompute the XB CVI incrementally.\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.param_inc-Tuple{CONN,Int64,Int64,Int64}","page":"Index","title":"AdaptiveResonance.param_inc","text":"param_inc(cvi::CONN, p::Int64, p2::Int64, label::Int64)\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.patch_orientation_color-Tuple{Array,Array}","page":"Index","title":"AdaptiveResonance.patch_orientation_color","text":"patch_orientation_color(z::Array, image::Array)\n\nARTSCENE Stage 6: Create patch feature vectors.\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.performance-Tuple{Array,Array}","page":"Index","title":"AdaptiveResonance.performance","text":"performance(y_hat::Array, y::Array)\n\nReturns the categorization performance of y_hat against y.\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.similarity-Tuple{String,GNFA,String,Array,Real}","page":"Index","title":"AdaptiveResonance.similarity","text":"similarity(method::String, F2::GNFA, field_name::String, sample::Array, gamma_ref::Real)\n\nCompute the similarity metric depending on method with explicit comparisons for the field name.\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.stopping_conditions-Tuple{DAM}","page":"Index","title":"AdaptiveResonance.stopping_conditions","text":"stopping_conditions(art::DAM)\n\nStopping conditions for Default ARTMAP, checked at the end of every epoch.\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.stopping_conditions-Tuple{DDVFA}","page":"Index","title":"AdaptiveResonance.stopping_conditions","text":"stopping_conditions(art::DDVFA)\n\nStopping conditions for Distributed Dual Vigilance Fuzzy ARTMAP.\n\nReturns true if there is no change in weights during the epoch or the maxmimum epochs has been reached.\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.stopping_conditions-Tuple{GNFA}","page":"Index","title":"AdaptiveResonance.stopping_conditions","text":"stopping_conditions(art::GNFA)\n\nStopping conditions for a GNFA module.\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.stopping_conditions-Tuple{SFAM}","page":"Index","title":"AdaptiveResonance.stopping_conditions","text":"stopping_conditions(art::SFAM)\n\nStopping conditions for Simple Fuzzy ARTMAP, checked at the end of every epoch.\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.surround_kernel-NTuple{5,Int64}","page":"Index","title":"AdaptiveResonance.surround_kernel","text":"surround_kernel(i::Int, j::Int, p::Int, q::Int, scale::Int)\n\nSurround kernel S function for ARTSCENE Stage 2\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.train!-Tuple{DAM,Array,Array}","page":"Index","title":"AdaptiveResonance.train!","text":"train!(art::DAM, x::Array, y::Array ; preprocessed=false)\n\nTrains a Default ARTMAP learner in a supervised manner.\n\nExamples\n\njulia> x, y = load_data()\njulia> art = DAM()\nDAM\n    opts: opts_DAM\n    ...\njulia> train!(art, x, y)\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.train!-Tuple{DDVFA,Array}","page":"Index","title":"AdaptiveResonance.train!","text":"train!(art::DDVFA, x::Array ; preprocessed=false)\n\nTrain the DDVFA model on the data.\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.train!-Tuple{GNFA,Array}","page":"Index","title":"AdaptiveResonance.train!","text":"train!(art::GNFA, x::Array ; y::Array=[])\n\nTrains a GNFA learner with dataset 'x' and optional labels 'y'\n\nExamples\n\njulia> my_GNFA = GNFA()\nGNFA\n    opts: opts_GNFA\n    ...\njulia> x = load_data()\njulia> train!(my_GNFA, x)\n\n\n\n\n\n","category":"method"},{"location":"man/full-index/#AdaptiveResonance.train!-Tuple{SFAM,Array,Array}","page":"Index","title":"AdaptiveResonance.train!","text":"train!(art::SFAM, x::Array, y::Array ; preprocessed=false)\n\nTrains a Simple Fuzzy ARTMAP learner in a supervised manner.\n\nExamples\n\njulia> x, y = load_data()\njulia> art = SFAM()\nSFAM\n    opts: opts_SFAM\n    ...\njulia> train!(art, x, y)\n\n\n\n\n\n","category":"method"},{"location":"man/guide/#Package-Guide","page":"Guide","title":"Package Guide","text":"","category":"section"},{"location":"man/guide/#Installation","page":"Guide","title":"Installation","text":"","category":"section"},{"location":"man/guide/","page":"Guide","title":"Guide","text":"The AdaptiveResonance package can be installed using the Julia package manager. From the Julia REPL, type ']' to enter the Pkg REPL mode and run","category":"page"},{"location":"man/guide/","page":"Guide","title":"Guide","text":"pkg> add AdaptiveResonance","category":"page"},{"location":"man/guide/","page":"Guide","title":"Guide","text":"Alternatively, it can be added to/ your environment in a script with","category":"page"},{"location":"man/guide/","page":"Guide","title":"Guide","text":"using Pkg\nPkg.add(\"AdaptiveResonance\")","category":"page"},{"location":"man/guide/#Usage","page":"Guide","title":"Usage","text":"","category":"section"},{"location":"man/guide/","page":"Guide","title":"Guide","text":"The AdaptiveResonance package is built upon ART modules that contain all of the state information during training and inference. The ART modules are driven by options, which are themselves mutable keyword argument structs from the Parameters.jl package.","category":"page"},{"location":"man/guide/#ART-Modules","page":"Guide","title":"ART Modules","text":"","category":"section"},{"location":"man/guide/","page":"Guide","title":"Guide","text":"The AdaptiveResonance package is designed for maximum flexibility for scientific research, even though this may come at the cost of learning instability if misused. Because of the diversity of ART modules, the package is structured around instantiating separate modules and using them for training and inference. Due to this diversity, each module has its own options struct with keyword arguments. These options have default values driven by standards in their respective literatures, so the ART modules may be used immediately without any customization. Furthermore, these options are mutable, so they may be modified before module instantiation, before training, or even after training.","category":"page"},{"location":"man/guide/","page":"Guide","title":"Guide","text":"For example, you can get going with the default options by creating an ART module with the default constructor:","category":"page"},{"location":"man/guide/","page":"Guide","title":"Guide","text":"my_art = DDVFA()","category":"page"},{"location":"man/guide/","page":"Guide","title":"Guide","text":"If you want to change the parameters before construction, you can create an options struct, modify it, then instantiate your ART module with it:","category":"page"},{"location":"man/guide/","page":"Guide","title":"Guide","text":"my_art_opts = opts_DDVFA()\nmy_art_opts.gamma = 3\nmy_art = DDVFA(my_art_opts)","category":"page"},{"location":"man/guide/","page":"Guide","title":"Guide","text":"The options are objects from the Parameters.jl project,","category":"page"},{"location":"man/guide/","page":"Guide","title":"Guide","text":"You can even modify the parameters on the fly after the ART module has been instantiated by directly modifying the options within the module:","category":"page"},{"location":"man/guide/","page":"Guide","title":"Guide","text":"my_art = DDVFA()\nmy_art.opts.gamma = 3","category":"page"},{"location":"man/examples/#Examples","page":"Examples","title":"Examples","text":"","category":"section"},{"location":"man/examples/","page":"Examples","title":"Examples","text":"Here are some helpful examples of the usage of the module.","category":"page"},{"location":"man/examples/#Example-1:","page":"Examples","title":"Example 1:","text":"","category":"section"},{"location":"man/examples/","page":"Examples","title":"Examples","text":"art = DDVFA()","category":"page"},{"location":"man/examples/#Example-2:","page":"Examples","title":"Example 2:","text":"","category":"section"},{"location":"man/examples/","page":"Examples","title":"Examples","text":"art = SFAM()","category":"page"},{"location":"#AdaptiveResonance.jl","page":"Home","title":"AdaptiveResonance.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The Package Guide provides a tutorial explaining how to get started using Documenter.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Some examples of packages using Documenter can be found on the Examples page.","category":"page"},{"location":"","page":"Home","title":"Home","text":"See the Index for the complete list of documented functions and types.","category":"page"},{"location":"#Manual-Outline","page":"Home","title":"Manual Outline","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Pages = [\n    \"man/guide.md\",\n    \"man/examples.md\",\n    \"man/contributing.md\",\n    \"man/full-index.md\",\n]\nDepth = 1","category":"page"}]
}
