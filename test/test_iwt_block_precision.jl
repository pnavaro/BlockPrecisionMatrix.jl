@testset "IWT block precision" begin

using UnicodePlots

import BlockPrecisionMatrix: structure_cov, cov_simu, generate_data
import BlockPrecisionMatrix: permute, permute_scad, iwt_block_precision

p = 10 
n = 500
b = 3
rng = MersenneTwister(12)
blocs_on  = [[1,3]]

covmat, premat, data, blocks = generate_data(rng, p, n, b, blocs_on)

display(heatmap(covmat, title="covmat"))
display(heatmap(premat, title="premat"))

@time pval = iwt_block_precision(rng, data, blocks; B=1000)

display(heatmap(pval, title="pval"))

@test true

end
