# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 12/01/2014

module Solver_

export Solver, selectAction


abstract Solver


selectAction(alg::Solver) = error("$(typeof(alg)) does not implement selectAction()")

end


