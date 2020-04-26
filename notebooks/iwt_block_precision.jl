# -*- coding: utf-8 -*-
using CategoricalArrays
using Random
using LinearAlgebra
using Distributions
using GLMNet
using InvertedIndices
using NCVREG
using ProgressMeter
using InvertedIndices
using Statistics
using UnicodePlots


function RotationMatrix(p :: Int64)

    if p == 1 return 1.0*Matrix(I,1,1) end

    P = zeros(Float64, (p,p)) 

    if p == 2
        u = rand()
        sqrt_1_u2 = sqrt(1 - u*u)
        P[1,1] = sqrt_1_u2;    # P[0, 0] = sqrt(1 - u^2) = cos(theta)
        P[2,1] = u;            # P[1, 0] = u = sin(theta)
        P[1,2] = -u;           # P[0, 1] = -u = -sin(theta)
        P[2,2] = sqrt_1_u2;    # P[1, 1] = sqrt(1 - u^2) = cos(theta)
        return P
    end

    Q, R = qr(rand(p, p))
    P = collect(Q)

    if p % 2 == 0

        if first(P) < 0 
           P .*= -1
        end

        E = 1.0 .* Matrix(I, p, p)
        for k in 2:p
            for j in 2:k, i in 1:p
                E[i,j] = P[i,j]
            end
            if det(E) < 0
                P[:,k] .*= -1
            end
        end 
    end
  
    return P

end

"""
    structure_cov(p, blocs_on; b = 10, seed = p)

# Fonction pour definir les indices des blocs

  - p : dimension des donnees
  - b : nombre de blocs
  - blocs_on : liste des indices des couples de blocs "allumes"
"""
function structure_cov(rng :: AbstractRNG,
                       p :: Int, 
                       b :: Int, 
                       blocs_on :: Vector{Vector{Int}} )
  
  blocs = sample(rng, 3:(p-2), (b-1), replace=false)
  sort!(blocs)
  blocs = [1, blocs..., p]
  indblocs = UnitRange{Int64}[]
  push!(indblocs, 1:blocs[2])

  for i in 2:(length(blocs)-1) 
    push!(indblocs, (blocs[i]+1):blocs[i+1])
  end
  
  matBlocs = zeros(Int64, (p, p))

  for l in indblocs
      matBlocs[l, l] .= 1
  end
  
  for bloc in blocs_on
      for i in indblocs[bloc[1]]
          for j in indblocs[bloc[2]]
              matBlocs[i, j] = 1
              matBlocs[j, i] = 1
          end
      end
  end
  
  return blocs, indblocs, matBlocs

end


"""
    cov_simu(blocs, indblocs, blocs_on, D)

# fonction pour simuler des matrices de covariances avec blocs
  - blocs    : indice des intervalles separant les blocs (sortie de StructureCov)
  - indblocs : liste des indices des blocs (sortie de StructureCov)
  - blocs_on  : liste des indices des couples de blocs "allumes"
  - D        : vecteur des valeurs propres de la matrice a simuler
"""
function cov_simu(blocs, indblocs, blocs_on, D)
  
  b = length(blocs) - 1
  p = sum(map(length,indblocs))
  P = zeros(Float64, (p, p))
  
  # matrices de rotation pour les blocs "allumes"
  for i in eachindex(blocs_on)
      B1 = indblocs[blocs_on[i][1]]
      B2 = indblocs[blocs_on[i][2]]
      nB = length(B1) + length(B2)
      P[[B1..., B2...], [B1..., B2...]] .= RotationMatrix(nB)
  end
  
  # matrices de rotation pour les autre blocs centraux (non allumes)
  for i in setdiff(1:b, Iterators.flatten(blocs_on))
      P[indblocs[i], indblocs[i]] .= RotationMatrix(length(indblocs[i]))
  end
  
  for i in 1:length(blocs_on)
    P[indblocs[blocs_on[i][1]] , indblocs[blocs_on[i][2]]] .= 0
  end

  P .= transpose(P)
  
  # Matrice de covariance
  CovMat = Hermitian(P' * Diagonal(D) * P)
  
  # Matrice de precision
  PreMat = P' * Diagonal(1 ./ D) * P
  
  return collect(CovMat), PreMat

end

# +

function generate_data(rng, p, n, b, blocs_on )

    blocs, indblocs = structure_cov(rng, p, b, blocs_on)

    D = rand(rng, Uniform(1e-4, 1e-2), p)

    covmat, premat   = cov_simu(blocs, indblocs, blocs_on, D)
    
    d = MvNormal(covmat)
    p_part = map( length,  indblocs)
    blocks  = vcat([repeat([i], j) for (i,j) in zip(1:b, p_part)]...)
    data = rand!(rng, d, zeros(Float64,(p, n)))
    covmat, premat, transpose(data), blocks

end

# -


# +
function permute(rng, x, n) 
   result = copy(x)
   result[randperm(rng, n), :]
end

function permute_conditional(rng :: AbstractRNG, Y, Z)
    row_y, col_y = size(Y)
    row_z, col_z = size(Z)
    
    @assert row_y == row_z
    
    fitted = similar(Y)
    for j in 1:col_y
        y = Y[:,j]
        λ = 2*sqrt(var(Y[:,1])*log(col_z)/row_z)
        beta = NCVREG.coef(NCVREG.SCAD(Z, y, [λ]))  
        fitted[:,j] .= vec(hcat(ones(row_z),Z) * beta)
    end
    
    residuals = Y .- fitted
    
    permutation = randperm(rng, row_y)
        
    return fitted .+ residuals[permutation,:]
end


# +
function stat_test(block1_perm, block2, X, points_x, points_y)
    
    X[:, points_x] .= block1_perm

    n, p = size(X)
    Tprec = zeros(p, p)
    TprecStd = zeros(p, p)

    betahat = zeros(Float64,(p-1,p))
    reshat  = copy(X)
    y = zeros(Float64, n)
    
    @inbounds for k in 1:p
        x = view(X, :, Not(k))
        y .= X[:, k]
        λ = [2*sqrt(var(y)*log(p)/n)]
        fitreg = glmnet(x, y, lambda = λ, standardize=false)
        y .= vec(GLMNet.predict(fitreg,x))
        betahat[:,k] .= vec(fitreg.betas)
        for i in eachindex(y)
           reshat[i,k] = X[i, k] - y[i]
        end
    end
    
    rtilde = cov(reshat) .* (n-1) ./ n
    rhat   = rtilde
    
    @inbounds for i in 1:p-1
        for j in (i+1):p
            rhat[i,j] = -(rtilde[i,j]+rtilde[i,i]*betahat[i,j]+rtilde[j,j]*betahat[j-1,i])
            rhat[j,i] = rhat[i,j]
        end
    end
    
    @inbounds for i in eachindex(Tprec, rhat)
        Tprec[i] = 1 / rhat[i]
    end

    TprecStd .= Tprec

    for i in 1:(p-1)
        for j in (i+1):p
            Tprec[i,j] = Tprec[j,i] = rhat[i,j]/(rhat[i,i]*rhat[j,j])
            thetahatij = (1+(betahat[i,j]^2  * rhat[i,i]/rhat[j,j]))/(n*rhat[i,i]*rhat[j,j])   
            TprecStd[i,j] = TprecStd[j,i] = Tprec[i,j]/sqrt(thetahatij)
        end
    end

    submat = TprecStd[points_x, points_y]
    return sum(submat).^2
    
end
# -

function iwt_block_precision(data, blocks; B=1000)
    
    estimation = :SCAD
    n, p  = size(data)
    nblocks = length(levels(CategoricalArray(blocks)))
    # tests on rectangles
    pval_array = zeros(Float64,(2,p,p))
    corrected_pval = zeros(Float64, (p,p))
    responsible_test = Array{String,2}(undef,p,p)
    ntests_blocks = zeros(Float64,(p,p))
    zeromatrix = zeros(Float64,(p,p))
    testmatrix = zeros(Float64,(p,p))
    index = zeros(Int,(p,p))
    corrected_pval_temp = zeros(Float64,(p,p))
    Tperm_tmp = zeros(Float64, B)

    # x coordinate starting point and length on x axis of the rectangle
    @showprogress 1 for ix in 2:nblocks, lx in 0:(nblocks-ix)

        # FIRST BLOCK

        index_x = ix:(ix+lx) # index first block
        points_x = findall(x -> x in index_x, blocks) # coefficients in block index.x
        data_B1 = data[:,points_x] # data of the first block
        data_B1_array = Iterators.repeated(data_B1, B)

        # y coordinate starting point. stops before the diagonal and length on y axis of the rectangle
        for iy in 1:(ix-1), ly in 0:(ix-iy-1)

            # SECOND BLOCK

            index_y = iy:(iy+ly) # index second block
            points_y = findall( x -> x in index_y, blocks)
            data_B2 = data[:,points_y]
            index_complement = collect(filter(index -> !(index in vcat(index_x,index_y)), 1:nblocks))
            points_complement = findall( x -> x in index_complement, blocks)
            data_complement = data[:,points_complement]

            # permuted data of the first block
            ncol = size(data_complement)[2]

            if ncol > 0
                data_B1_perm = [permute_conditional(rng, B1, data_complement) for B1 in data_B1_array];
            else
                data_B1_perm = [permute(rng, B1, n) for B1 in data_B1_array];
            end

            fill!(testmatrix, 0.0) 
            testmatrix[points_x,points_y] .= 1
            ntests_blocks .+= testmatrix

            heatmap(ntests_blocks, title="Blocks: $index_x - $index_y")
            heatmap(testmatrix,    title="Blocks: $index_x - $index_y")
            
            T0_tmp = stat_test(data_B1,data_B2,data,points_x,points_y)

            fill!(Tperm_tmp, 0.0)

            Threads.@threads for i in eachindex(Tperm_tmp)
                @inbounds Tperm_tmp[i] = stat_test(data_B1_perm[i], data_B2, data, points_x, points_y)
            end

            pval_tmp = mean(Tperm_tmp .>= T0_tmp)

            corrected_pval_temp[points_x,points_y] .= pval_tmp
            corrected_pval_temp[points_y,points_x] .= pval_tmp # symetrization

            # old adjusted p-value
            # pval_array[1,:,:] .= corrected_pval 
            # p-value resulting from the test of this block
            # pval_array[2,:,:] .= corrected_pval_temp 
            # maximization for updating the adjusted p-value

            for i in 1:p, j in 1:p
                corrected_pval[i,j] = max(corrected_pval[i,j], corrected_pval_temp[i,j])
            end

            #for i in 1:p, j in 1:p 
            #    index[i,j] = findmax(pval_array[:,i,j])[2] 
            #end

            #responsible_test[ index .== 2] .= "$(paste(index_x))-$(paste(index_y))"
            
        end
    end
    
    corrected_pval
end

p = 20 
n = 1000
b = 5
rng = MersenneTwister(42)
blocs_on  = [[1,3]]

covmat, premat, data, blocks = generate_data(rng, p, n, b, blocs_on)

heatmap(covmat)

heatmap(premat)

@show blocks

@time res = iwt_block_precision(data, blocks; B=1000)

heatmap(res)


