#!/usr/bin/env julia
using CSVFiles, DataFrames, LinearAlgebra
using Clustering # kmeans
meta_data = DataFrame(load(File(format"TSV", "data/meta_data.tsv.gz")));
categorical!(meta_data, :dataset);
data_mat = DataFrame(load(File(format"TSV", "data/pcs.tsv.gz")));

vars_use = "dataset"

function run_harmony(data_mat, meta_data, vars_use, sigma, 
    tau, block_size, max_iter_harmony, max_iter_cluster, epsilon_cluster,
    epsilon_harmony)
    println("Sigma is $sigma")
    N = size(meta_data, 1)
    n_clust = minimum([N ÷ 30, 100])
    theta = 2.
    lamb = 0.1
    
    sigma = fill(sigma, N)

    labels = meta_data[!, Symbol(vars_use)]
    categories = size(levels(labels),1)
    
    phi = fill(0., (categories, N))
    
    for (index, value) in enumerate(levels(labels))
        phi[index,:] = (value .== labels)
    end
    
    N_b = sum(phi, dims=2)
    Pr_b = N_b / N
    
    B_vec = [categories]
    
    println("Bveec is $B_vec")
    
    theta = fill(theta, size(B_vec,1))
    
    lamb = fill(lamb, size(B_vec,1))
    
    lamb_mat = Diagonal(pushfirst!(lamb, 0))
    
    padding = fill(1., N)
    
    phi_moe = vcat(padding', phi)
    
    harmony(data_mat, phi, phi_moe, Pr_b, sigma, theta, n_clust, lamb_mat)
    
end

function harmony(data_mat, phi, phi_moe, Pr_b, sigma, theta, n_clust, lamb_mat)
    max_iter_cluster = 200
    epsilon_cluster = 1e-5
    epsilon_harmony = 1e-4
    block_size = 0.05
    
    # normalize data_mat
    data_mat_cos = data_mat ./ (maximum.(eachcol(data_mat)))'
    
    # calculate norm
    # why not use LinearAlgebra.norm?
    data_mat_norm = mapslices(norm, convert(Matrix, data_mat_cos), dims=1)
    
    data_mat_cos = data_mat_cos ./ data_mat_norm

    M = convert(Matrix, data_mat_cos)

    d, n = size(data_mat)
    B = size(phi, 1)
    
    window_size = 3
    
    # initialize cluster and then harmonize
    println("Initializing cluster with n_clust: $n_clust")
    init_cluster(M, n_clust, sigma, Pr_b, phi)
    harmonize()
end

function init_cluster(M, num_clust, sigma, Pr_b, phi)
    # nb: kmeans works with $d x $n data matrix - hence the transpose.
    km = kmeans(M', num_clust, maxiter=10)
    
    @assert nclusters(km) == num_clust
    
    Y = km.centers'
    
    println("Normalizing...")
    # normalize
    Y_norm = mapslices(norm, convert(Matrix, Y), dims=1)
    Y = Y ./ Y_norm
    println("Begin cluster assignment")
    # assign cluster probabilities
    dist_mat = 2 * (1 .- (Y * M'))
    
    R = -dist_mat
    sigma = reshape(sigma, (length(sigma), 1));
    R = R ./ sigma';
    max = maximum.(eachcol(R));
    R = R .- reshape(max, (length(max), 1))'; 
    R = exp.(R)
    summy = sum.(eachcol(R))
    R = R ./ reshape(summy, (length(summy), 1))';
    println("Assignment done!")
    # (3) batch diversity statistics
    E = sum.(eachrow(R)) .* Pr_b'
    O = R * phi'

    println("E shape ", size(E))
    println("O shape ", size(O))

    compute_objective()
    
    
end


function harmonize(iter_harmony=10)
    for i in 1:10
        println("This is iteration $i out of $iter_harmony")
        
        # step 1: clustering
        cluster()
        
        # step 2: regress out covariates
        Z_cos, Z_corr, W, Phi_Rk = moe_correct_ridge(Z_orig, Z_cos, Z_corr,
        R, W, K, Phi_Rk, Phi_moe, lamb)
        
        # step 3: check for convergence
        #if check_convergence()
        #    println("Harmony converged after $i iteration(s)")
        #    break
    end
end

function cluster()
end

function moe_correct_ridge()
end

function check_convergence()
end

function compute_objective()
    kmeans_error = 5
end

function safe_entropy(x)
    f(j) = log(j)
    y = x .* map.(f, x)
    if any(isnan.(y))
        y[isnan.(y)] .= 0.
    end
    return y
end

run_harmony(data_mat, meta_data, vars_use, 0.1, 0, 0.05, 10, 200, 1e-5, 1e-4)
