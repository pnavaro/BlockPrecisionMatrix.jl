push!(LOAD_PATH,"../src/")

using Documenter
using PrecisionMatrix

makedocs(
    repo="https://github.com/pnavaro/PrecisionMatrix/blob/{commit}{path}#L{line}",
    sitename = "PrecisionMatrix",
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true",
        canonical="https://pnavaro.github.io/PrecisionMatrix"),
    modules = [PrecisionMatrix],
    pages=[
        "Home" => "index.md",
    ]
)

deploydocs(;
    repo="github.com/pnavaro/PrecisionMatrix",
)
