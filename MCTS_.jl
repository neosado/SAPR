# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 11/04/2014

# Monte-Carlo Tree Search
module MCTS_

export MCTS, selectAction, reinitialize, initialize


using Solver_


import Solver_.selectAction


abstract MCTS <: Solver


selectAction(alg::MCTS) = error("$(typeof(alg)) does not implement selectAction()")
reinitialize(alg::MCTS) = error("$(typeof(alg)) does not implement reinitialize()")
initialize(alg::MCTS) = error("$(typeof(alg)) does not implement initialize()")

end


