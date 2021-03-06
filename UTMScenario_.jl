# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 03/09/2015

VERSION >= v"0.4" && __precompile__(false)


module UTMScenario_

export UTMScenario, UTMScenarioParams, UTMScenarioState
export updateState, isEndState


using UAV_


type UTMScenarioParams

    scenario::Union{Int64, Void}

    x::Float64
    y::Float64

    dt::Int64

    cell_towers::Union{Vector{Vector{Float64}}, Void}

    landing_bases::Union{Vector{Vector{Float64}}, Void}

    jamming_time::Float64
    jamming_center::Vector{Float64}
    jamming_radius::Float64

    UAVs::Union{Vector{UAV}, Void}
    nUAV::Int64

    sa_dist::Float64

    # for nav1
    loc_err_sigma::Float64
    loc_err_bound::Float64

    heading_err_sigma::Float64
    velocity_err_sigma::Float64

    bMCTS::Bool


    function UTMScenarioParams(scenario::Union{Int64, Void} = nothing)

        self = new()

        self.scenario = scenario

        self.x = 0. # ft
        self.y = 0. # ft

        self.dt = 0 # seconds

        self.cell_towers = nothing

        self.landing_bases = nothing

        self.jamming_time = Inf
        self.jamming_center = [0., 0.]
        self.jamming_radius = 0.

        self.UAVs = nothing
        self.nUAV = 0

        self.sa_dist = 0. # ft

        self.loc_err_sigma = 0.     # ft
        self.loc_err_bound = 0.     # ft

        self.heading_err_sigma = 0.     # degree
        self.velocity_err_sigma = 0.    # ft/s

        self.bMCTS = false

        return self
    end
end


type UTMScenario

    scenario::Union{Int64, Void}

    x::Float64
    y::Float64

    dt::Int64

    cell_towers::Union{Vector{Vector{Float64}}, Void}

    landing_bases::Union{Vector{Vector{Float64}}, Void}

    jamming_time::Float64
    jamming_center::Vector{Float64}
    jamming_radius::Float64

    UAVs::Union{Vector{UAV}, Void}
    nUAV::Int64

    sa_dist::Float64

    # for nav1
    loc_err_sigma::Float64
    loc_err_bound::Float64

    heading_err_sigma::Float64
    velocity_err_sigma::Float64

    bMCTS::Bool


    function UTMScenario(params::UTMScenarioParams)

        self = new()

        self.scenario = params.scenario

        self.x = params.x
        self.y = params.y

        self.dt = params.dt

        self.cell_towers = params.cell_towers

        self.landing_bases = params.landing_bases

        self.jamming_time = params.jamming_time
        self.jamming_center = params.jamming_center
        self.jamming_radius = params.jamming_radius

        self.UAVs = params.UAVs
        self.nUAV = params.nUAV

        for i = 1:self.nUAV
            uav = self.UAVs[i]
            uav.sc = self
        end

        self.sa_dist = params.sa_dist

        self.loc_err_sigma = params.loc_err_sigma
        self.loc_err_bound = params.loc_err_bound

        self.heading_err_sigma = params.heading_err_sigma
        self.velocity_err_sigma = params.velocity_err_sigma

        self.bMCTS = params.bMCTS

        return self
    end
end


type UTMScenarioState

    UAVStates::Vector{UAVState}


    function UTMScenarioState(sc::UTMScenario)

        self = new()
        self.UAVStates = UAVState[]

        for uav in sc.UAVs
            push!(self.UAVStates, UAVState(uav))
        end

        return self
    end
end


function updateState(sc::UTMScenario, sc_state::UTMScenarioState, t::Int64)

    if t > 0
        for i = 1:sc.nUAV
            UAV_.updateState(sc.UAVs[i], sc_state.UAVStates[i], t)
        end
    end
end


function isEndState(sc::UTMScenario, sc_state::UTMScenarioState; uav_indexes::Union{Int64, Vector{Int64}, Void} = nothing)

    end_flag = true

    if uav_indexes == nothing
        uav_indexes = 1:sc.nUAV
    end

    for i = uav_indexes
        end_flag &= UAV_.isEndState(sc.UAVs[i], sc_state.UAVStates[i])
    end

    return end_flag
end


end


