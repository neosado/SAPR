# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 08/04/2015

module UTMPlannerV1_

import Base.isequal, Base.hash, Base.copy
import Base.start, Base.done, Base.next

export UTMPlannerV1, UPState, UPAction, UPObservation, UPBelief, UPBeliefVector, UPBeliefParticles, History
#export UPStateIter, UPObservationIter
export Generative, nextState, observe, reward, isEnd, isFeasible, sampleBelief, updateBelief
#export tranProb, obsProb
export coord2grid, grid2coord


using POMDP_
using Base.Test
using Iterators
using Distributions
using JSON

using UAV_
using Scenario_


import POMDP_.Generative
import POMDP_.nextState
import POMDP_.observe
import POMDP_.reward
import POMDP_.isEnd
import POMDP_.isFeasible
import POMDP_.sampleBelief
import POMDP_.updateBelief
import POMDP_.tranProb
import POMDP_.obsProb


#
# UPState, UPAction, UPObservation, UPBelief
#

immutable UPState <: State

    location::(Int64, Int64)
    status::Symbol
    heading::Symbol
    t::Int64
end

type UPStateIter

    nx::Int64
    ny::Int64
    dt::Int64
    t_max::Int64

    function UPStateIter(nx, ny, dt, t_max)

        self = new()

        self.nx = nx
        self.ny = ny
        self.dt = dt
        self.t_max = t_max

        return self
    end
end

# TODO implement start, done, next


immutable UPAction <: Action

    action::Symbol
end


immutable UPObservation <: Observation

    location::(Int64, Int64)
end

type UPObservationIter

    nx::Int64
    ny::Int64

    function UPObservationIter(nx, ny)

        self = new()

        self.nx = nx
        self.ny = ny

        return self
    end
end

# XXX observations can be outside of the box
function start(iter::UPObservationIter)

    state = (1, 1)
end

function done(iter::UPObservationIter, state)

    x, y = state

    if y > iter.ny
        return true
    end

    return false
end

function next(iter::UPObservationIter, state)

    x, y = state

    item = UPObservation((x, y))

    if x + 1 > iter.nx
        x = 1
        y += 1
    else
        x += 1
    end

    return item, (x, y)
end


abstract UPBelief <: Belief

immutable UPBeliefVector <: UPBelief
    belief::Dict{UPState, Float64}
end

immutable UPBeliefParticles <: UPBelief
    particles::Vector{UPState}
end


#
# scenario
#

function generateScenario()

    # parameters

    params = ScenarioParams()

    params.x = 5010.    # ft
    params.y = 5010.    # ft

    params.dt = 1       # seconds

    params.cell_towers = Vector{Float64}[[160., 200.], [2100., 4900.], [4800., 500.]]

    params.landing_bases = Vector{Float64}[[1200., 800.], [4000., 1000.], [2000., 3500.]]

    params.sa_dist = 30.    # ft

    # nav1
    params.loc_err_sigma = 200
    params.loc_err_bound = params.loc_err_sigma

    UAVs = {{"route" => {[100., 2500.], [2000., 2000.], [3600., 3000.], [5500., 3000.]}, "v" => 40},
    {"route" => {[3200., 100.], [4400., 1800.], [3800., 3900.], [4400., 5500.]}, "v" => 23},
    {"route" => {[4900., 2200.], [1600., 1200.], [-500., 900.]}, "v" => 25},
    {"route" => {[4000., 4900.], [2000., 3000.], [1500., -500.]}, "v" => 50},
    {"route" => {[400., 4900.], [2700., 4200.], [5500., 4000.]}, "v" => 40}}

    UAVList = UAV[]

    for i = 1:length(UAVs)
        uav_info = UAVs[i]

        uav = UAV()

        uav.index = i

        route = uav_info["route"]

        uav.start_loc = route[1]

        if length(route) > 2
            uav.waypoints = Vector{Float64}[]
            uav.nwaypoint = length(route) - 2

            for j = 2:length(route)-1
                push!(uav.waypoints, route[j])
            end
        end

        uav.end_loc = route[end]

        if haskey(uav_info, "v")
            uav.velocity = uav_info["v"]
        else
            uav.velocity = 40.
        end

        uav.velocity_min = 20.
        uav.velocity_max = 60.

        # GPS_INS, deadreckoning or radiolocation
        if i == 1
            uav.navigation = :nav1
        else
            uav.navigation = :GPS_INS
        end

        # nav1
        uav.IMU_gyr_sigma = 0.1

        push!(UAVList, uav)
    end

    params.UAVs = UAV[]
    params.nUAV = length(UAVList)

    for i = 1:params.nUAV
        push!(params.UAVs, UAVList[i])
    end

    sc = Scenario(params)

    sc_state = ScenarioState(sc)

    return sc, sc_state
end


#
# UTMPlannerV1
#

type UTMPlannerV1 <: POMDP

    seed::Union(Int64, Nothing)

    sc::Scenario
    sc_state::ScenarioState

    dt::Int64           # seconds

    cell_len::Float64   # ft
    n::Int64

    states::UPStateIter
    nState::Int64

    actions::Vector{UPAction}
    nAction::Int64

    observations::UPObservationIter
    nObservation::Int64

    obsProbMat::Array{Float64, 2}

    UAVStates::Any

    reward_functype::Symbol
    reward_norm_const::Float64


    function UTMPlannerV1(; seed::Union(Int64, Nothing) = nothing)

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

        self.sc, self.sc_state = generateScenario()
        self.sc.seed = seed

        self.dt = 5

        self.cell_len = 30.
        @assert self.sc.x == self.sc.y && self.sc.x % self.cell_len == 0

        self.n = int64(self.sc.x / self.cell_len)
        @assert self.n % 2 == 1

        # XXX think about how to handle time in state
        self.states = UPStateIter(self.n, self.n, self.dt, 237.)
        self.nState = self.n * self.n

        self.actions = [UPAction(:None_), UPAction(:Waypoint1), UPAction(:Waypoint2), UPAction(:End), UPAction(:Base1), UPAction(:Base2), UPAction(:Base3)]
        self.nAction = length(self.actions)

        self.observations = UPObservationIter(self.n, self.n)
        self.nObservation = self.n * self.n

        self.obsProbMat = zeros(self.n, self.n)
        D = MvNormal(zeros(2), self.sc.loc_err_sigma)
        for j = 1:self.n
            for i = 1:self.n
                self.obsProbMat[i, j] = pdf(D, [(i - 1) * 30, (j - 1) * 30])
            end
        end
        self.obsProbMat /= ((sum(self.obsProbMat) - sum(self.obsProbMat[:, 1])) * 4 + self.obsProbMat[1, 1])

        self.UAVStates = JSON.parsefile("UTMPlannerV1.json")

        self.reward_functype = :type2
        self.reward_norm_const = 1000.

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
        xg = int64(ceil(x / up.cell_len))
    end

    if y == 0.
        yg = 1
    else
        yg = int64(ceil(y / up.cell_len))
    end

    return (xg, yg)
end


function grid2coord(up::UTMPlannerV1, grid::(Int64, Int64))

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
        up.sc_state.UAVStates[i].index = up.UAVStates[t + 1][i]["index"]
        up.sc_state.UAVStates[i].status = up.UAVStates[t + 1][i]["status"]
        up.sc_state.UAVStates[i].curr_loc = copy(up.UAVStates[t + 1][i]["curr_loc"])
        up.sc_state.UAVStates[i].past_locs = deepcopy(up.UAVStates[t + 1][i]["past_locs"])
        up.sc_state.UAVStates[i].heading = up.UAVStates[t + 1][i]["heading"]
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

    r = 0

    if a.action != :None_
        atype, aindex, aloc = convertHeading(uav, a.action)
        htype, hindex, hloc = convertHeading(uav, s.heading)

        dist = 0.

        if atype == :waypoint
            for i = hindex:aindex-1
                dist += computeDist(uav_state.curr_loc, uav.waypoints[aindex], uav.waypoints[i])
            end

            r += -int64(ceil(dist / up.cell_len))

        elseif atype == :end_
            for i = hindex:uav.nwaypoint
                dist += computeDist(uav_state.curr_loc, uav.end_loc, uav.waypoints[i])
            end

            r += -int64(ceil(dist / up.cell_len))

        elseif atype == :base
            c1 = 1.
            c2 = 0.
            r += -int64(ceil(norm(aloc - uav_state.curr_loc) / up.cell_len * c1 + c2))

        end
    end

    if a.action != :None_
        updateHeading(uav, uav_state, a.action)
    end

    for i = 1:up.dt
        if uav_state.status == :flying
            for k = 2:up.sc.nUAV
                state__ = up.sc_state.UAVStates[k]

                if  state__.status == :flying
                    if norm(state__.curr_loc - uav_state.curr_loc) < up.sc.sa_dist
                        r += -1000
                    end
                end
            end
        end

        UAV_.updateState(uav, uav_state, t)

        t += up.sc.dt

        if UAV_.isEndState(uav, uav_state)
            break
        end

        restoreUAVStates(up, t)
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
function reward(up::UTMPlannerV1, s::UPState, a::UPAction)

    restoreUAVStates(up, s.t)

    uav = up.sc.UAVs[1]
    uav_state = up.sc_state.UAVStates[1]
    uav_state.index = 1
    uav_state.status = s.status
    uav_state.heading = s.heading
    uav_state.curr_loc = grid2coord(up, s.location)

    r = 0

    if a.action != :None_
        atype, aindex, aloc = convertHeading(uav, a.action)
        htype, hindex, hloc = convertHeading(uav, s.heading)

        dist = 0.

        if atype == :waypoint
            for i = hindex:aindex-1
                dist += computeDist(uav_state.curr_loc, uav.waypoints[aindex], uav.waypoints[i])
            end

            r += -int64(ceil(dist / up.cell_len))

        elseif atype == :end_
            for i = hindex:uav.nwaypoint
                dist += computeDist(uav_state.curr_loc, uav.end_loc, uav.waypoints[i])
            end

            r += -int64(ceil(dist / up.cell_len))

        elseif atype == :base
            c1 = 1.
            c2 = 0.
            r += -int64(ceil(norm(aloc - uav_state.curr_loc) / up.cell_len * c1 + c2))

        end
    end

    if uav_state.status == :flying
        for i = 2:up.sc.nUAV
            state__ = up.sc_state.UAVStates[i]

            if  state__.status == :flying
                if norm(state__.curr_loc - uav_state.curr_loc) < up.sc.sa_dist
                    r += -1000
                end
            end
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

function hash(s::UPState, h::Uint64 = zero(Uint64))

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

function hash(a::UPAction, h::Uint64 = zero(Uint64))

    return hash(a.action, h)
end


function isequal(o1::UPObservation, o2::UPObservation)

    return isequal(o1.location, o2.location)
end

function ==(o1::UPObservation, o2::UPObservation)

    return (o1.location == o2.location)
end

function hash(o::UPObservation, h::Uint64 = zero(Uint64))

    return hash(o.location, h)
end


function isequal(k1::(History, UPAction), k2::(History, UPAction))

    return isequal(k1[1], k2[1]) && isequal(k1[2], k2[2])
end

function ==(k1::(History, UPAction), k2::(History, UPAction))

    return (k1[1] == k2[1]) && (k1[2] == k2[2])
end

function hash(k::(History, UPAction), h::Uint64 = zero(Uint64))

    return hash(k[2], hash(k[1], h))
end


end


