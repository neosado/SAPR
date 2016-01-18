# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 08/04/2015

VERSION >= v"0.4" && __precompile__(false)


module UTMPlannerV1_

import Base: isequal, ==, hash, copy, string
import Base: start, done, next

export UTMPlannerV1, UPState, UPAction, UPObservation, UPBelief, UPBeliefVector, UPBeliefParticles, History
export UPStateIter, UPObservationIter
export nextState, observe, reward, Generative, isEnd, isFeasible, sampleBelief, updateBelief
export tranProb, obsProb
export coord2grid, grid2coord


using POMDP_
using Iterators
using Distributions
using Base.Test

using UAV_
using UTMScenario_
using UTMScenarioGenerator_


import POMDP_: nextState, observe, reward, Generative, isEnd, isFeasible, sampleBelief, updateBelief, tranProb, obsProb


#
# UPState, UPAction, UPObservation, UPBelief
#

immutable UPState <: State

    location::Tuple{Int64, Int64}
    status::Symbol
    heading::Symbol
    t::Int64
end

function string(s::UPState)
    return "(" * string(s.location) * "," * string(s.status) * "," * string(s.heading) * "," * string(s.t) * ")"
end

type UPStateIter

    function UPStateIter()

        error("UPStateIter has not been implemented yet")
    end
end


immutable UPAction <: Action

    action::Symbol
end

function string(a::UPAction)

    return string(a.action)
end


immutable UPObservation <: Observation

    location::Tuple{Int64, Int64}
end

function string(o::Observation)

    return string(o.location)
end

type UPObservationIter

    function UPObservationIter(nx, ny)

        error("UPObservationIter has not been implemented yet")
    end
end


abstract UPBelief <: Belief

immutable UPBeliefVector <: UPBelief
    belief::Dict{UPState, Float64}
end

immutable UPBeliefParticles <: UPBelief
    particles::Vector{UPState}
end


#
# UTMPlannerV1
#

type UTMPlannerV1 <: POMDP

    seed::Union{Int64, Void}

    sc::UTMScenario
    sc_state::UTMScenarioState

    dt::Int64           # seconds

    cell_len::Float64   # ft
    n::Int64

    states::Union{UPStateIter, Void}
    nStates::Int64

    actions::Vector{UPAction}
    nActions::Int64

    observations::Union{UPObservationIter, Void}
    nObservation::Int64

    obsProbMat::Array{Float64, 2}

    UAVStates::Any

    reward_min::Int64
    reward_max::Int64

    reward_norm_const::Float64


    function UTMPlannerV1(; seed::Union{Int64, Void} = nothing, scenario_number::Union{Int64, Void} = nothing, Scenarios = nothing)

        self = new()

        if seed == nothing
            self.seed = round(Int64, time())
        else
            self.seed = seed
        end

        # Note: can't create rng since Distributions.jl does not support rng argument in its rand()
        srand(self.seed)

        self.sc, self.sc_state, self.UAVStates, _ = generateScenario(scenario_number, navigation = :nav1, Scenarios = Scenarios)

        self.dt = 5

        self.cell_len = 30.
        @assert self.sc.x == self.sc.y && self.sc.x % self.cell_len == 0

        self.n = round(Int64, self.sc.x / self.cell_len)
        @assert self.n % 2 == 1

        self.states = nothing
        self.nStates = 0

        if self.sc.UAVs[1].nwaypoints > 2
            self.actions = [UPAction(:None_), UPAction(:Waypoint1), UPAction(:Waypoint2), UPAction(:Waypoint3), UPAction(:Base1), UPAction(:Base2), UPAction(:Base3)]
        else
            self.actions = [UPAction(:None_), UPAction(:Waypoint1), UPAction(:Waypoint2), UPAction(:End), UPAction(:Base1), UPAction(:Base2), UPAction(:Base3)]
        end
        self.nActions = length(self.actions)

        self.observations = nothing
        self.nObservation = 0

        self.obsProbMat = zeros(self.n, self.n)
        D = MvNormal(zeros(2), self.sc.loc_err_sigma)
        for j = 1:self.n
            for i = 1:self.n
                self.obsProbMat[i, j] = pdf(D, [(i - 1) * self.cell_len, (j - 1) * self.cell_len]) * (self.cell_len * self.cell_len)
            end
        end

        self.reward_min = -10000
        self.reward_max = 0

        self.reward_norm_const = 1.

        return self
    end
end


# P(s' | s, a)
function tranProb(up::UTMPlannerV1, s::UPState, a::UPAction, s_::UPState)

    error("tranProb has not been implemented yet")
end


# P(o | s', a)
function obsProb(up::UTMPlannerV1, s_::UPState, a::UPAction, o::UPObservation)

    x_, y_ = s_.location
    x_obs, y_obs = o.location

    return up.obsProbMat[abs(x_obs - x_) + 1, abs(y_obs - y_) + 1]
end


function coord2grid(up::UTMPlannerV1, coord::Vector{Float64})

    x = coord[1]
    y = coord[2]

    if x == 0.
        xg = 1
    else
        xg = round(Int64, ceil(x / up.cell_len))
    end

    if y == 0.
        yg = 1
    else
        yg = round(Int64, ceil(y / up.cell_len))
    end

    return (xg, yg)
end


function grid2coord(up::UTMPlannerV1, grid::Tuple{Int64, Int64})

    xg, yg = grid

    x = up.cell_len / 2 + (xg - 1 ) * up.cell_len
    y = up.cell_len / 2 + (yg - 1 ) * up.cell_len

    return [x, y]
end


function restoreUAVStates(up::UTMPlannerV1, t::Int64)

    if t > length(up.UAVStates) - 1
        t = length(up.UAVStates) - 1
    end

    for i = 2:up.sc.nUAV
        up.sc_state.UAVStates[i] = up.UAVStates[t + 1][i]
    end
end


computeDist(p1::Vector{Float64}, p2::Vector{Float64}, p::Vector{Float64}) = abs((p2[1] - p1[1]) * (p1[2] - p[2]) - (p1[1] - p[1]) * (p2[2] - p1[2])) / sqrt((p2[1] - p1[1])^2 + (p2[2] - p1[2])^2)


function Generative(up::UTMPlannerV1, s::UPState, a::UPAction)

    t = s.t

    restoreUAVStates(up, t)

    uav = up.sc.UAVs[1]
    uav_state = up.sc_state.UAVStates[1]
    uav_state.index = 1
    uav_state.status = s.status
    uav_state.heading = s.heading
    uav_state.curr_loc = grid2coord(up, s.location)

    if a.action != :None_
        updateHeading(uav, uav_state, a.action)
    end

    r = 0

    bMAC = false

    htype, hindex, hloc = convertHeading(uav, uav_state.heading)
    htype_prev = htype
    hindex_prev = hindex

    for i = 1:up.dt
        if uav_state.status == :flying
            if htype == :base
                r += -1
            end

            if !bMAC
                for k = 2:up.sc.nUAV
                    state__ = up.sc_state.UAVStates[k]

                    if  state__.status == :flying
                        if norm(state__.curr_loc - uav_state.curr_loc) < up.sc.sa_dist
                            bMAC = true
                            break
                        end
                    end
                end
            end
        end

        UAV_.updateState(uav, uav_state, t)

        if uav_state.status == :flying
            htype, hindex, hloc = convertHeading(uav, uav_state.heading)

            if (htype_prev == :waypoint && htype == :waypoint && hindex == hindex_prev + 1) || (htype_prev == :waypoint && htype == :end_)
                r += 100

                htype_prev = htype
                hindex_prev = hindex
            end
        end

        t += up.sc.dt

        if UAV_.isEndState(uav, uav_state)
            break
        end

        restoreUAVStates(up, t)
    end

    if bMAC
        r += -10000
    end

    s_ = UPState(coord2grid(up, uav_state.curr_loc), uav_state.status, uav_state.heading, t)

    if up.sc.loc_err_sigma == 0
        o = UPObservation(s_.location)
    else
        o = UPObservation(coord2grid(up, rand(MvNormal(uav_state.curr_loc, up.sc.loc_err_sigma))))
    end

    return s_, o, r
end


# s' ~ P(S | s, a)
function nextState(up::UTMPlannerV1, s::UPState, a::UPAction)

    restoreUAVStates(up, s.t)

    uav = up.sc.UAVs[1]
    uav_state = up.sc_state.UAVStates[1]
    uav_state.index = 1
    uav_state.status = s.status
    uav_state.heading = s.heading
    uav_state.curr_loc = grid2coord(up, s.location)

    if a.action != :None_
        updateHeading(uav, uav_state, a.action)
    end

    # XXX trajectory error caused by discretization
    UAV_.updateState(uav, uav_state, s.t)

    s_ = UPState(coord2grid(up, uav_state.curr_loc), uav_state.status, uav_state.heading, s.t + up.sc.dt)

    return s_
end


# o ~ P(O | s', a)
function observe(up::UTMPlannerV1, s_::UPState, a::UPAction)

    if up.sc.loc_err_sigma == 0
        o = UPObservation(s_.location)
    else
        # FIXME observation should follow obsProbMat
        o = UPObservation(coord2grid(up, rand(MvNormal(grid2coord(up, s_.location), up.sc.loc_err_sigma))))
    end

    return o
end


# R(s, a)
function reward(up::UTMPlannerV1, s::UPState, a::UPAction, s_::UPAction)

    restoreUAVStates(up, s.t)

    uav = up.sc.UAVs[1]
    uav_state = up.sc_state.UAVStates[1]
    uav_state.index = 1
    uav_state.status = s.status
    uav_state.heading = s.heading
    uav_state.curr_loc = grid2coord(up, s.location)

    r = 0

    htype, hindex, hloc = convertHeading(uav, s.heading)
    htype_, hindex_, hloc_ = convertHeading(uav, s_.heading)

    if uav_state.status == :flying
        if htype == :base
            r += -1
        end

        for i = 2:up.sc.nUAV
            state__ = up.sc_state.UAVStates[i]

            if  state__.status == :flying
                if norm(state__.curr_loc - uav_state.curr_loc) < up.sc.sa_dist
                    r += -10000
                end
            end
        end

        if (htype == :waypoint && htype_ == :waypoint && hindex_ == hindex + 1) || (htype == :waypoint && htype_ == :end_)
            r += 100
        end
    end

    return r
end


function isEnd(up::UTMPlannerV1, s::UPState)

    restoreUAVStates(up, s.t)

    uav = up.sc.UAVs[1]
    uav_state = up.sc_state.UAVStates[1]
    uav_state.index = 1
    uav_state.status = s.status
    uav_state.heading = s.heading
    uav_state.curr_loc = grid2coord(up, s.location)

    return UAV_.isEndState(uav, uav_state)
end


function isFeasible(up::UTMPlannerV1, s::UPState, a::UPAction)

    if a.action == :None_
        return true
    end

    htype, hindex, hloc = convertHeading(up.sc.UAVs[1], s.heading)
    atype, aindex, aloc = convertHeading(up.sc.UAVs[1], a.action)

    if htype == atype && hindex == aindex
        return false
    end

    if htype == :waypoint
        if atype == :waypoint
            if aindex >= hindex
                return true
            end
        else
            return true
        end

    elseif htype == :base
        if atype == :base
            return true
        end

    elseif htype == :end_
        if atype == :base || atype == :end_
            return true
        end

    end

    return false
end


# s ~ b
function sampleBelief(up::UTMPlannerV1, b::UPBeliefVector)

    error("sampleBelief has not been implemented yet")
end

function sampleBelief(up::UTMPlannerV1, b::UPBeliefParticles)

    s = b.particles[rand(1:length(b.particles))]

    return copy(s)
end


# b' = B(b, a, o)
function updateBelief(up::UTMPlannerV1, b::UPBeliefVector, a::UPAction, o::UPObservation)

    error("updateBelief has not been implemented yet")
end

function updateBelief(up::UTMPlannerV1, b::UPBeliefParticles)

    return b
end


function isequal(s1::UPState, s2::UPState)

    return isequal(s1.location, s2.location) && isequal(s1.status, s2.status) && isequal(s1.heading, s2.heading) && isequal(s1.t, s2.t)
end

function ==(s1::UPState, s2::UPState)

    return (s1.location == s2.location) && (s1.status == s2.status) && (s1.heading == s2.heading) && (s1.t == s2.t)
end

function hash(s::UPState, h::UInt64 = zero(UInt64))

    return hash(s.t, hash(s.heading, hash(s.status, hash(s.location, h))))
end

function copy(s::UPState)

    return UPState(s.location, s.status, s.heading, s.t)
end


function isequal(a1::UPAction, a2::UPAction)

    return isequal(a1.action, a2.action)
end

function ==(a1::UPAction, a2::UPAction)

    return (a1.action == a2.action)
end

function hash(a::UPAction, h::UInt64 = zero(UInt64))

    return hash(a.action, h)
end


function isequal(o1::UPObservation, o2::UPObservation)

    return isequal(o1.location, o2.location)
end

function ==(o1::UPObservation, o2::UPObservation)

    return (o1.location == o2.location)
end

function hash(o::UPObservation, h::UInt64 = zero(UInt64))

    return hash(o.location, h)
end


function isequal(k1::Tuple{History, UPAction}, k2::Tuple{History, UPAction})

    return isequal(k1[1], k2[1]) && isequal(k1[2], k2[2])
end

function ==(k1::Tuple{History, UPAction}, k2::Tuple{History, UPAction})

    return (k1[1] == k2[1]) && (k1[2] == k2[2])
end

function hash(k::Tuple{History, UPAction}, h::UInt64 = zero(UInt64))

    return hash(k[2], hash(k[1], h))
end


end


