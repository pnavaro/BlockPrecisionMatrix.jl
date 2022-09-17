push!(LOAD_PATH,"../src/")

using Documenter
using BlockPrecisionMatrix

makedocs(
    repo="https://github.com/pnavaro/BlockPrecisionMatrix.jl/blob/{commit}{path}#L{line}",
    sitename = "BlockPrecisionMatrix.jl",
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true",
        canonical="https://pnavaro.github.io/BlockPrecisionMatrix.jl"),
    modules = [BlockPrecisionMatrix],
    pages=[
        "Home" => "index.md",
    ]
)

deploydocs(;
    repo="github.com/pnavaro/BlockPrecisionMatrix.jl",
)
