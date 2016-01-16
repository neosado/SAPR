# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 11/04/2014

VERSION >= v"0.4" && __precompile__(false)


# Partially Observable Markov Decision Process
module POMDP_

import Base: isequal, ==, hash

export POMDP, State, Action, Observation, Belief, History
export nextState, observe, reward, Generative, isEnd, isFeasible, sampleBelief, updateBelief
export tranProb, obsProb


abstract POMDP
abstract State
abstract Action
abstract Observation
abstract Belief


nextState(pm::POMDP) = error("$(typeof(pm)) does not implement nextState()")
observe(pm::POMDP) = error("$(typeof(pm)) does not implement observe()")
reward(pm::POMDP) = error("$(typeof(pm)) does not implement reward()")
Generative(pm::POMDP) = error("$(typeof(pm)) does not implement Generative()")
isEnd(pm::POMDP) = error("$(typeof(pm)) does not implement isEnd()")
isFeasible(pm::POMDP) = error("$(typeof(pm)) does not implement isFeasible()")
sampleBelief(pm::POMDP) = error("$(typeof(pm)) does not implement sampleBelief()")
updateBelief(pm::POMDP) = error("$(typeof(pm)) does not implement updateBelief()")
tranProb(pm::POMDP) = error("$(typeof(pm)) does not implement tranProb()")
obsProb(pm::POMDP) = error("$(typeof(pm)) does not implement obsProb()")


type History

    history::Vector{Any}


    function History(history = [])

        self = new()
        self.history = history

        return self
    end
end

function isequal(h1::History, h2::History)

    if length(h1.history) != length(h2.history)
        return false
    else
        return reduce(&, [isequal(e1, e2) for (e1, e2) in zip(h1.history, h2.history)])
    end
end

function ==(h1::History, h2::History)

    if length(h1.history) != length(h2.history)
        return false
    else
        return reduce(&, [e1 == e2 for (e1, e2) in zip(h1.history, h2.history)])
    end
end

function hash(hist::History, h::UInt64 = zero(UInt64))

    h = hash(nothing, h)

    for h_ in hist.history
        h = hash(h_, h)
    end

    return h
end


end


