push!(LOAD_PATH,"../src/")

using Documenter
using PrecisionMatrix

makedocs(
    sitename = "PrecisionMatrix",
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    modules = [PrecisionMatrix]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
