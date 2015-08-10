# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 11/04/2014

module Visualizer_

export Visualizer, visInit, visUpdate, updateAnimation, saveAnimation


abstract Visualizer


visInit(vis::Visualizer) = error("$(typeof(vis)) does not implement visInit()")
visUpdate(vis::Visualizer) = error("$(typeof(vis)) does not implement visUpdate()")
updateAnimation(vis::Visualizer) = error("$(typeof(vis)) does not implement updateAnimation()")
saveAnimation(vis::Visualizer) = error("$(typeof(vis)) does not implement saveAnimation()")

end


