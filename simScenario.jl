# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 03/09/2015

using UAV_
using Scenario_
using UTMVisualizer_


function generateParameters(model::ASCIIString)

    params = ScenarioParams()

    if model == "v0.1"
        params.x = 5000.    # ft
        params.y = 5000.    # ft

        params.dt = 1.      # seconds

        params.cell_towers = Vector{Float64}[[160., 200.], [2100., 4900.], [4800., 500.]]

        params.landing_bases = Vector{Float64}[[1200., 800.], [4000., 1000.], [2000., 3500.]]

        params.sa_dist = 500. # ft

        # nav1
        params.loc_err_sigma = 200
        params.loc_err_bound = params.loc_err_sigma
    end

    return params
end


function generateUAVList(UAVs)

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
        uav.navigation = :GPS_INS

        # nav1
        uav.IMU_gyr_sigma = 0.1

        push!(UAVList, uav)
    end

    return UAVList
end


function translateUAVs(UAVs, x, y)

    UAVs_ = Any[]

    for uav_info in UAVs
        uav_info_ = Any[]

        for waypoint in uav_info
            push!(uav_info_, [waypoint[1] + x, waypoint[2] + y])
        end

        push!(UAVs_, uav_info_)
    end

    return UAVs_
end


function generateUAVs(model::ASCIIString)

    if model == "v0.1"
        UAVs = {{"route" => {[100., 2500.], [2000., 2000.], [3600., 3000.], [5500., 3000.]}, "v" => 40},
                {"route" => {[3200., 100.], [4400., 1800.], [3800., 3900.], [4400., 5500.]}, "v" => 23},
                {"route" => {[4900., 2200.], [1600., 1200.], [-500., 900.]}, "v" => 25},
                {"route" => {[4000., 4900.], [2000., 3000.], [1500., -500.]}, "v" => 50},
                {"route" => {[400., 4900.], [2700., 4200.], [5500., 4000.]}, "v" => 40}}
    end

    return generateUAVList(UAVs)
end


function generateScenario(model::ASCIIString; uav_indexes::Union(Int64, Vector{Int64}, Nothing) = nothing, navigations::Union(Symbol, Vector{Symbol}, Nothing) = nothing)

    params = generateParameters(model)

    UAVList = generateUAVs(model)

    params.UAVs = UAV[]
    params.nUAV = length(UAVList)

    for i = 1:params.nUAV
        uav = UAVList[i]

        if uav_indexes != nothing
            if i in uav_indexes
                if typeof(navigations) == Symbol
                    uav.navigation = navigations
                else
                    uav.navigation = navigations[findfirst(uav_indexes .== i)]
                end
            end
        end

        push!(params.UAVs, uav)
    end

    sc = Scenario(params)

    sc_state = ScenarioState(sc)

    return sc, sc_state
end


function simulate(sc::Scenario, sc_state::ScenarioState; draw::Bool = false, wait::Bool = false, uav_indexes::Union(Int64, Vector{Int64}, Nothing) = nothing, headings::Union(Symbol, Vector{Symbol}, Nothing) = nothing)

    if uav_indexes != nothing
        for uav_index = uav_indexes
            if typeof(headings) == Symbol
                heading = headings
            else
                heading = headings[findfirst(uav_indexes .== uav_index)]
            end

            updateHeading(sc.UAVs[uav_index], sc_state.UAVStates[uav_index], heading)
        end
    end

    if draw
        vis = UTMVisualizer(wait = wait)

        visInit(vis, sc, sc_state)
        visUpdate(vis, sc, sc_state)
        updateAnimation(vis)
    end

    t = 0

    sa_violation = zeros(Bool, sc.nUAV, sc.nUAV)
    sa_violation_count = 0

    while !isEndState(sc, sc_state, uav_indexes = uav_indexes)
        updateState(sc, sc_state, t)

        for i = 1:sc.nUAV-1
            for j = i+1:sc.nUAV
                state1 = sc_state.UAVStates[i]
                state2 = sc_state.UAVStates[j]

                if (state1.status == :flying || state1.status == :back_to_base) && (state2.status == :flying || state2.status == :back_to_base)
                    if norm(state1.curr_loc - state2.curr_loc) < sc.sa_dist
                        if !sa_violation[i, j]
                            sa_violation[i, j] = true
                            sa_violation_count += 1
                        end
                    else
                        if sa_violation[i, j]
                            sa_violation[i, j] = false
                        end
                    end
                end
            end
        end

        if draw
            visInit(vis, sc, sc_state)
            visUpdate(vis, sc, sc_state, t)
            updateAnimation(vis)
        end

        t += 1
    end

    if draw
        println("# of SA violations: ", sa_violation_count)

        saveAnimation(vis, repeat = true)
    end

    return sa_violation_count
end


if false
    srand(uint(time()))

    sc, sc_state = generateScenario("v0.1")

    simulate(sc, sc_state, draw = true)
end


if false
    srand(uint(time()))

    sc, sc_state = generateScenario("v0.1", uav_indexes = 1, navigations = :nav1)

    simulate(sc, sc_state, draw = true, wait = false, uav_indexes = 1, headings = :Waypoint1)
end


if false
    srand(uint(time()))

    N = 1000
    RE_threshold = 0.01

    va = Float64[]
    y = 0.

    n = 1
    while true
        sc, sc_state = generateScenario("v0.1", uav_indexes = 1, navigations = :nav1)

        x = simulate(sc, sc_state, uav_indexes = 1, headings = :Waypoint1)
        y += (x - y) / n
        push!(va, y)
        
        if n % 100 == 0
            if std(va) / va[end] < RE_threshold
                break
            end
        end

        if n == N
            break
        end

        n += 1
    end

    println("n: ", n, ", mean: ", va[end], ", std: ", std(va), ", RE: ", std(va) / va[end])
end


