module SpinAdaptedSecondQuantization

export SASQ
const SASQ = SpinAdaptedSecondQuantization

using DataStructures
using Permutations

"""
    Constraints = SortedDict{Int,Type}

Type alias for container of MO-Index constraints
"""
const Constraints = SortedDict{Int,Type}

include("orbital_spaces.jl")
include("spin.jl")
include("kronecker_delta.jl")
include("operator.jl")
include("tensor.jl")
include("term.jl")
include("expression.jl")

include("ket.jl")
include("wick_theorem.jl")

include("code_generation.jl")

include("code_generator.jl")

include("tensor_replacements.jl")

include("desymmetrization.jl")

include("precompile.jl")

end # module SpinAdaptedSecondQuantization
