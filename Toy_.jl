# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 08/25/2015

module Toy_

import Base: isequal, hash, copy

export Toy, TYState, TYAction, TYObservation, TYBelief, TYBeliefVector, TYBeliefParticles, History
export nextState, observe, reward, Generative, isEnd, isFeasible, sampleBelief, updateBelief


using POMDP_
using Base.Test


import POMDP_: nextState, observe, reward, Generative, isEnd, isFeasible, sampleBelief, updateBelief


immutable TYState <: State

    state::Symbol
end

immutable TYAction <: Action

    action::Symbol
end

immutable TYObservation <: Observation

    observation::Symbol
end

abstract TYBelief <: Belief

immutable TYBeliefVector <: TYBelief
    belief::Dict{TYState, Float64}
end

immutable TYBeliefParticles <: TYBelief
    particles::Vector{TYState}
end


type Toy <: POMDP

    seed::Union(Int64, Nothing)

    states::Vector{TYState}
    nStates::Int64

    actions::Vector{TYAction}
    nActions::Int64

    observations::Vector{TYObservation}
    nObservation::Int64

    reward_norm_const::Float64


    function Toy(; seed::Union(Int64, Nothing) = nothing)

        self = new()

        if seed != nothing
            if seed != 0
                self.seed = seed
            else
                self.seed = int64(time())
            end

            srand(self.seed)

        else
            self.seed = nothing

        end

        self.states = TYState[]
        self.nStates = 7
        for i = 1:self.nStates
            push!(self.states, TYState(symbol("S$i")))
        end

        self.actions = [TYAction(:Left), TYAction(:Right)]
        self.nActions = length(self.actions)

        self.observations = TYObservation[]
        self.nObservation = self.nStates
        for i = 1:self.nObservation
            push!(self.observations, TYObservation(symbol("S$i")))
        end

        self.reward_norm_const = 1000.

        return self
    end
end


# s', o, r ~ G(s, a)
function Generative(ty::Toy, s::TYState, a::TYAction)

    s_ = nextState(ty, s, a)
    o = observe(ty, s_, a)
    r = reward(ty, s, a)

    return s_, o, r
end
        

# s' ~ P(S | s, a)
function nextState(ty::Toy, s::TYState, a::TYAction)

    if s.state == :S1
        if a.action == :Left
            state_ = :S2
        elseif a.action == :Right
            state_ = :S3
        end

    elseif s.state == :S2
        if a.action == :Left
            state_ = :S4
        elseif a.action == :Right
            state_ = :S5
        end

    elseif s.state == :S3
        if a.action == :Left
            state_ = :S6
        elseif a.action == :Right
            state_ = :S7
        end

    end

    s_ = TYState(state_)

    return s_
end


# o ~ P(O | s', a)
function observe(ty::Toy, s_::TYState, a::TYAction)

    #o = TYObservation(symbol(s_.state)
    o = TYObservation(symbol(string(s_.state) * "_" * string(rand(1:typemax(Int16)))))

    return o
end


# R(s, a)
function reward(ty::Toy, s::TYState, a::TYAction)

    if s.state == :S1
        if a.action == :Left
            r = 0.
        elseif a.action == :Right
            r = -20.
        end

    elseif s.state == :S2
        if a.action == :Left
            r = 0.
        elseif a.action == :Right
            r = -800.
        end

    elseif s.state == :S3
        if a.action == :Left
            r = -10.
        elseif a.action == :Right
            r = -1000.
        end

    end

    return r
end


function isEnd(ty::Toy, s::TYState)

    if s.state == :S4 || s.state == :S5 || s.state == :S6 || s.state == :S7
        return true
    else
        return false
    end
end


function isFeasible(ty::Toy, s::TYState, a::TYAction)

    return true
end


# s ~ b
function sampleBelief(ty::Toy, b::TYBeliefVector)

    error("sampleBelief has not been implemented yet")
end

function sampleBelief(ty::Toy, b::TYBeliefParticles)

    s = b.particles[rand(1:length(b.particles))]

    return copy(s)
end


# b' = B(b, a, o)
function updateBelief(ty::Toy, b::TYBeliefVector, a::TYAction, o::TYObservation)

    error("updateBelief has not been implemented yet")
end

function updateBelief(ty::Toy, b::TYBeliefParticles)

    return b
end


function isequal(s1::TYState, s2::TYState)

    return isequal(s1.state, s2.state)
end

function ==(s1::TYState, s2::TYState)

    return (s1.state== s2.state)
end

function hash(s::TYState, h::Uint64 = zero(Uint64))

    return hash(s.state, h)
end

function copy(s::TYState)

    return TYState(s.state)
end


function isequal(a1::TYAction, a2::TYAction)

    return isequal(a1.action, a2.action)
end

function ==(a1::TYAction, a2::TYAction)

    return (a1.action == a2.action)
end

function hash(a::TYAction, h::Uint64 = zero(Uint64))

    return hash(a.action, h)
end


function isequal(o1::TYObservation, o2::TYObservation)

    return isequal(o1.observation, o2.observation)
end

function ==(o1::TYObservation, o2::TYObservation)

    return (o1.observation == o2.observation)
end

function hash(o::TYObservation, h::Uint64 = zero(Uint64))

    return hash(o.observation, h)
end


function isequal(k1::(History, TYAction), k2::(History, TYAction))

    return isequal(k1[1], k2[1]) && isequal(k1[2], k2[2])
end

function ==(k1::(History, TYAction), k2::(History, TYAction))

    return (k1[1] == k2[1]) && (k1[2] == k2[2])
end

function hash(k::(History, TYAction), h::Uint64 = zero(Uint64))

    return hash(k[2], hash(k[1], h))
end


end


