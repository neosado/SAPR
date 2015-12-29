# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 08/26/2015

module UTMScenarioGenerator_

export generateScenario, loadScenarios, simulateScenario


using UAV_
using Scenario_
using UTMVisualizer_
using Util

using JLD


function generateScenarioWithParams(params::ScenarioParams, UAVInfo; navigation::Symbol = :GPS_INS, v_def::Float64 = 0., v_min::Float64 = 0., v_max::Float64 = 0.)

    UAVList = UAV[]

    for i = 1:length(UAVInfo)
        uav_info = UAVInfo[i]["uav_info"]

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
            uav.velocity = v_def
        end

        uav.velocity_min = v_min
        uav.velocity_max = v_max

        # GPS_INS, deadreckoning or radiolocation
        if i == 1
            uav.navigation = navigation
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

    for i = 1:sc.nUAV
        if haskey(UAVInfo[i], "uav_state")
            uav_state = UAVInfo[i]["uav_state"]
            heading = uav_state["heading"]
            curr_loc = uav_state["curr_loc"]

            sc_state.UAVStates[i].curr_loc = curr_loc

            push!(sc_state.UAVStates[i].past_locs, sc.UAVs[i].start_loc)
            for j = 1:heading-2
                push!(sc_state.UAVStates[i].past_locs, sc.UAVs[i].waypoints[j])
            end
            push!(sc_state.UAVStates[i].past_locs, curr_loc)

            if heading == sc.UAVs[i].nwaypoint + 2
                sc_state.UAVStates[i].heading = symbol("End")
            else
                sc_state.UAVStates[i].heading = symbol("Waypoint" * string(heading - 1))
            end
        end
    end

    return sc, sc_state
end


function scenario_1()

    # parameters
    params = ScenarioParams()

    params.x = 5010.    # ft
    params.y = 5010.    # ft

    params.dt = 1       # seconds

    params.cell_towers = Vector{Float64}[[160., 200.], [2100., 4900.], [4800., 500.]]

    params.landing_bases = Vector{Float64}[[1200., 800.], [4000., 1000.], [2000., 3500.]]

    #params.sa_dist = 30.    # ft
    params.sa_dist = 500.    # ft

    # nav1
    params.loc_err_sigma = 200.     # ft
    params.loc_err_bound = params.loc_err_sigma     # ft

    params.heading_err_sigma = 0.   # degree
    params.velocity_err_sigma = 0.  # ft/s

    UAVInfo = Any[]

    push!(UAVInfo, Dict{ASCIIString, Any}("uav_info" => Dict{ASCIIString, Any}("route" => Vector{Float64}[[100., 2500.], [2000., 2000.], [3600., 3000.], [5500., 3000.]], "v" => 40)))
    push!(UAVInfo, Dict{ASCIIString, Any}("uav_info" => Dict{ASCIIString, Any}("route" => Vector{Float64}[[3200., 100.], [4400., 1800.], [3800., 3900.], [4400., 5500.]], "v" => 23)))
    push!(UAVInfo, Dict{ASCIIString, Any}("uav_info" => Dict{ASCIIString, Any}("route" => Vector{Float64}[[4900., 2200.], [1600., 1200.], [-500., 900.]], "v" => 25)))
    push!(UAVInfo, Dict{ASCIIString, Any}("uav_info" => Dict{ASCIIString, Any}("route" => Vector{Float64}[[4000., 4900.], [2000., 3000.], [1500., -500.]], "v" => 50)))
    push!(UAVInfo, Dict{ASCIIString, Any}("uav_info" => Dict{ASCIIString, Any}("route" => Vector{Float64}[[400., 4900.], [2700., 4200.], [5500., 4000.]], "v" => 40)))

    sc, sc_state = generateScenarioWithParams(params, UAVInfo, v_def = 40., v_min = 20., v_max = 60.)

    params.UAVs = nothing

    return sc, sc_state, UAVInfo, params
end


function checkLocationInside(x, y, loc)

    bInside = false

    if loc[1] >= 0 && loc[1] <= x && loc[2] >= 0 && loc[2] <= y
        bInside = true
    end

    return bInside
end


function generateBases(rng::AbstractRNG; x::Float64 = 0., y::Float64 = 0., margin::Float64 = 0., nbase::Int64 = 0, min_dist::Float64 = 0.)

    bases = Vector{Float64}[]

    for i = 1:nbase
        x_ = nothing
        y_ = nothing

        while true
            bOk = true

            x_ = Float64(round(Int64, margin + rand(rng) * (x - 2 * margin)))
            y_ = Float64(round(Int64, margin + rand(rng) * (y - 2 * margin)))

            for base_loc in bases
                if norm([x_, y_] - base_loc) < min_dist
                    bOk = false
                    break
                end
            end

            if bOk
                break
            end
        end

        push!(bases, round([x_, y_]))
    end

    return bases
end


function generateUAV(rng::AbstractRNG; x::Float64 = 0., y::Float64 = 0., margin::Float64 = 0., dist_mean::Float64 = 0., dist_noise::Float64 = 0., angle_noise::Float64 = 0., v_mean::Float64 = 0., v_min::Float64 = 0., v_max::Float64 = 0.)

    start_side = randi(rng, 1:4)

    if start_side == 1
        x_ = -dist_mean / 2
        y_ = Float64(round(Int64, margin + rand(rng) * (y - 2 * margin)))
        u = [1., 0.]
    elseif start_side == 2
        x_ = Float64(round(Int64, margin + rand(rng) * (x - 2 * margin)))
        y_ = -dist_mean / 2
        u = [0., 1.]
    elseif start_side == 3
        x_ = x + dist_mean / 2
        y_ = Float64(round(Int64, margin + rand(rng) * (y - 2 * margin)))
        u = [-1., 0.]
    elseif start_side == 4
        x_ = Float64(round(Int64, margin + rand(rng) * (x - 2 * margin)))
        y_ = y + dist_mean / 2
        u = [0., -1.]
    end

    start_loc = round([x_, y_])

    route = nothing

    while true
        route = Vector{Float64}[]

        push!(route, start_loc)

        curr_loc = start_loc
        u_ = u

        bInside = false

        while true
            angle = randn(rng) * angle_noise / 2

            if angle < -angle_noise
                angle = -angle_noise
            elseif angle > angle_noise
                angle = angle_noise
            end

            d = dist_mean + randn(rng) * dist_mean * dist_noise / 2

            if d < dist_mean * (1 - dist_noise)
                d = dist_mean * (1- dist_noise)
            elseif d > dist_mean * (1 + dist_noise)
                d = dist_mean * (1 + dist_noise)
            end

            u_ = [cosd(angle) -sind(angle); sind(angle) cosd(angle)] * u_

            next_loc = round(curr_loc + u_ * d)

            if !checkLocationInside(x, y, next_loc)
                push!(route, next_loc)
                break
            elseif next_loc[1] < margin || next_loc[1] > x - margin || next_loc[2] < margin || next_loc[2] > y - margin
                continue
            end

            push!(route, next_loc)

            if checkLocationInside(x, y, next_loc)
                bInside = true
            end

            curr_loc = next_loc
        end

        if bInside
            break
        end
    end

    v_noise = max(v_mean - v_min, v_max - v_mean)
    v = round(v_mean + randn(rng) * v_noise / 2)
    if v < v_min
        v = v_min
    elseif v > v_max
        v = v_max
    end

    uav_info = Dict{ASCIIString, Any}("route" => route, "v" => v)

    return uav_info
end


function getInitLocation(rng::AbstractRNG, params::ScenarioParams, route::Vector{Vector{Float64}}, rindex_noise::Int64)

    # assume that whole planned path is within the area except start and end points

    rindex = 1 + round(Int64, floor(abs(randn(rng)) * rindex_noise / 2))

    if rindex == 1
        rn = 0.

        while true
            curr_loc = route[rindex] + (route[rindex + 1] - route[rindex]) * rn

            if checkLocationInside(params.x, params.y, curr_loc)
                curr_loc += (route[rindex + 1] - route[rindex]) * (1 - rn) * rand(rng)
                break
            end

            rn += 0.1

            @assert rn <= 1.
        end

    elseif rindex > length(route) - 2
        rindex = length(route) - 1
        curr_loc = route[rindex]

    else
        curr_loc = route[rindex] + (route[rindex + 1] - route[rindex]) * rand(rng)

    end

    return round(curr_loc), rindex + 1
end


function getClosestPoint(sc::Scenario, sc_state::ScenarioState, uav_number::Int64)

    min_dist = Inf
    loc = nothing
    loc_ = nothing

    t = 0

    while !Scenario_.isEndState(sc, sc_state)
        Scenario_.updateState(sc, sc_state, t)

        state = sc_state.UAVStates[uav_number]
        state_ = sc_state.UAVStates[1]

        if state.status == :flying && state_.status == :flying
            if norm(state.curr_loc - state_.curr_loc) < min_dist
                min_dist = norm(state.curr_loc - state_.curr_loc)
                loc = state.curr_loc
                loc_ = state_.curr_loc
            end
        end

        t += 1
    end

    return loc, loc_, min_dist
end


function check_sa_violation(sc::Scenario, sc_state::ScenarioState; draw::Bool = false, wait::Bool = false)

    bViolation = false

    if draw
        vis = UTMVisualizer(wait = wait)

        visInit(vis, sc, sc_state)
        visUpdate(vis, sc, sc_state)
        updateAnimation(vis)
    end

    t = 0

    while !Scenario_.isEndState(sc, sc_state)
        Scenario_.updateState(sc, sc_state, t)

        state = sc_state.UAVStates[sc.nUAV]

        for i = 1:sc.nUAV-1
            state_ = sc_state.UAVStates[i]

            if state.status == :flying && state_.status == :flying
                if norm(state.curr_loc - state_.curr_loc) < sc.sa_dist
                    bViolation = true
                    break
                end
            end

            if bViolation
                break
            end
        end

        if draw
            visInit(vis, sc, sc_state)
            visUpdate(vis, sc, sc_state, t)
            updateAnimation(vis)
        end

        if bViolation
            break
        end

        t += 1
    end

    if draw
        readline()
        close(vis.fig)
    end

    return bViolation
end


function generateScenario_(seed::Int64)

    # should be cautious about other rand()s in Scenario_.updateState
    rng = MersenneTwister(seed)

    # parameters
    params = ScenarioParams()

    params.x = 5010.    # ft
    params.y = 5010.    # ft

    params.dt = 1       # seconds

    params.cell_towers = nothing

    params.landing_bases = generateBases(rng, x = params.x, y = params.y, margin = 100., nbase = 3, min_dist = 2500.)

    params.sa_dist = 500.    # ft

    # nav1
    params.loc_err_sigma = 200.     # ft
    params.loc_err_bound = params.loc_err_sigma     # ft

    params.heading_err_sigma = 0.   # degree
    params.velocity_err_sigma = 0.  # ft/s

    # parameters for generating scenario
    nUAV = 5
    min_route_points = 4
    rindex_noise = 2
    nNearbyUAV = 2
    sep_dist_margin = 20.
    sep_dist_margin_noise = 5.

    UAVInfo = Any[]

    for i = 1:nUAV
        while true
            uav_info = generateUAV(rng, x = params.x, y = params.y, margin = 100., dist_mean = 1500., dist_noise = 0.3, angle_noise = 45., v_mean = 40., v_min = 20., v_max = 60.)

            route = uav_info["route"]

            if length(route) < min_route_points
                continue
            end

            curr_loc, heading = getInitLocation(rng, params, route, rindex_noise)

            if i == 1
                push!(UAVInfo, Dict{ASCIIString, Any}("uav_info" => uav_info, "uav_state" => Dict{ASCIIString, Any}("heading" => heading, "curr_loc" => curr_loc)))
                break

            else
                UAVInfo_ = copy(UAVInfo)

                push!(UAVInfo_, Dict{ASCIIString, Any}("uav_info" => uav_info, "uav_state" => Dict{ASCIIString, Any}("heading" => heading, "curr_loc" => curr_loc)))

                sc, sc_state = generateScenarioWithParams(params, UAVInfo_; v_def = 40., v_min = 20., v_max = 60.)

                if i < 2 + nNearbyUAV
                    loc, loc_, dist = getClosestPoint(sc, sc_state, i)

                    if dist - sc.sa_dist > sep_dist_margin
                        dist_error = abs(randn(rng)) * sep_dist_margin_noise / 2
                        if dist_error < -sep_dist_margin_noise
                            dist_error = -sep_dist_margin_noise
                        elseif dist_error > sep_dist_margin_noise
                            dist_error = sep_dist_margin_noise
                        end

                        tran_vec = round((loc_ - loc) / norm(loc_ - loc) * (dist - sc.sa_dist - sep_dist_margin + dist_error))

                        for j = 1:length(route)
                            route[j] += tran_vec
                        end
                        UAVInfo_[i]["uav_state"]["curr_loc"] += tran_vec

                        nInsideWaypoints = 0
                        for j = 2:length(route)-1
                            if checkLocationInside(params.x, params.y, route[j])
                                nInsideWaypoints += 1
                            end
                        end

                        if nInsideWaypoints < min_route_points - 2
                            continue
                        end
                    end

                    sc, sc_state = generateScenarioWithParams(params, UAVInfo_; v_def = 40., v_min = 20., v_max = 60.)

                    if !check_sa_violation(sc, sc_state, draw = false, wait = false)
                        UAVInfo = UAVInfo_
                        break
                    end

                else
                    if !check_sa_violation(sc, sc_state, draw = false, wait = false)
                        UAVInfo = UAVInfo_
                        break
                    end

                end

            end
        end
    end

    sc, sc_state = generateScenarioWithParams(params, UAVInfo; v_def = 40., v_min = 20., v_max = 60.)

    params.UAVs = nothing

    return sc, sc_state, UAVInfo, params
end


function generateScenario(scenario_number::Union{Int64, Vector{Int64}, Void} = nothing; draw::Bool = false, wait::Bool = false, bSave::Bool = false, bAppend::Bool = true, navigation::Symbol = :GPS_INS, Scenarios = nothing)

    if Scenarios == nothing
        if isfile("Scenarios.jld")
            Scenarios = load("Scenarios.jld", "Scenarios")
        else
            Scenarios = Dict()
        end
    else
        Scenarios = Scenarios
    end

    if !bAppend
        Scenarios = Dict()
    end

    if scenario_number == nothing
        scenario_number = rand(1025:typemax(Int16))
    end

    bNewScenario = false

    for sn in scenario_number
        #println("scenario: ", sn)

        if !haskey(Scenarios, sn)
            if sn <= 1024
                sc, sc_state, UAVInfo, params = eval(symbol("scenario_" * string(sn)))()
            else
                @assert sn <= typemax(Int16)
                sc, sc_state, UAVInfo, params = generateScenario_(sn)
            end

            UAVStates = Any[]

            if draw
                vis = UTMVisualizer(wait = wait)

                visInit(vis, sc, sc_state)
                visUpdate(vis, sc, sc_state)
                updateAnimation(vis)
            end

            t = 0

            while !Scenario_.isEndState(sc, sc_state)
                Scenario_.updateState(sc, sc_state, t)

                push!(UAVStates, deepcopy(sc_state.UAVStates))

                if draw
                    visInit(vis, sc, sc_state)
                    visUpdate(vis, sc, sc_state, t)
                    updateAnimation(vis)
                end

                t += 1
            end

            if draw
                readline()
                close(vis.fig)
            end

            Scenarios[sn] = Dict{ASCIIString, Any}("UAVInfo" => UAVInfo, "UAVStates" => UAVStates, "params" => params)

            bNewScenario = true

        else
            UAVInfo = Scenarios[sn]["UAVInfo"]
            UAVStates = Scenarios[sn]["UAVStates"]
            params = Scenarios[sn]["params"]

        end

        if typeof(scenario_number) == Int64
            if bNewScenario && bSave
                save("Scenarios.jld", "Scenarios", Scenarios)
            end

            sc, sc_state = generateScenarioWithParams(params, UAVInfo, navigation = navigation; v_def = 40., v_min = 20., v_max = 60.)

            return sc, sc_state, UAVStates, sn
        end
    end

    if bNewScenario && bSave
        save("Scenarios.jld", "Scenarios", Scenarios)
    end
end


function loadScenarios()

    if isfile("Scenarios.jld")
        Scenarios = load("Scenarios.jld", "Scenarios")
    else
        Scenarios = nothing
    end
end


function simulateScenario(scenario_number::Union{Int64, Vector{Int64}, Void} = nothing; draw::Bool = true, wait::Bool = false, bSim::Bool = false, navigation::Symbol = :GPS_INS, Scenarios = nothing)

    if scenario_number == nothing
        scenario_number = rand(1025:typemax(Int16))
    end

    for sn in scenario_number
        println("scenario: ", sn)

        sc, sc_state, UAVStates, _ = generateScenario(sn, navigation = navigation, Scenarios = Scenarios)

        vis = UTMVisualizer(wait = wait)

        if draw
            visInit(vis, sc, sc_state)
            visUpdate(vis, sc, sc_state)
            updateAnimation(vis)
            if bSim && !wait
                readline()
            end
        end

        if bSim
            t = 0

            while !Scenario_.isEndState(sc, sc_state)
                Scenario_.updateState(sc, sc_state, t)

                if draw
                    visInit(vis, sc, sc_state)
                    visUpdate(vis, sc, sc_state, t)
                    updateAnimation(vis)
                end

                t += 1
            end
        end

        if draw
            #saveAnimation(vis, repeat = true)
            readline()
            close(vis.fig)
        end
    end
end


end


