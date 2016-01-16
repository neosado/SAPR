# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 03/09/2015

VERSION >= v"0.4" && __precompile__(false)


module UTMVisualizer_

export UTMVisualizer, visInit, visUpdate, updateAnimation, saveAnimation


using Visualizer_
using UTMScenario_

using PyCall
using PyPlot

@pyimport matplotlib.animation as ani
@pyimport matplotlib.patches as patch

import Visualizer_: visInit, visUpdate, updateAnimation, saveAnimation


type UTMVisualizer <: Visualizer

    fig::Union{Figure, Void}
    ax::Union{PyObject, Void}

    artists::Union{Vector{PyObject}, Void}

    ims::Vector{Any}

    wait::Bool

    uav_start_locs::Union{Vector{Vector{Float64}}, Void}


    function UTMVisualizer(;wait = false)

        self = new()

        self.fig = nothing
        self.ax = nothing

        self.artists = nothing

        self.ims = Any[]

        self.wait = wait

        self.uav_start_locs = nothing
        
        return self
    end
end


function visInit(vis::UTMVisualizer, sc::UTMScenario, sc_state::UTMScenarioState)

    if vis.fig == nothing
        fig = figure(facecolor = "white")

        ax = fig[:add_subplot](111)
        ax[:set_aspect]("equal")
        ax[:set_xlim](0, sc.x)
        ax[:set_ylim](0, sc.y)
        ax[:set_xticklabels]([])
        ax[:set_yticklabels]([])
        ax[:grid](true)
        ax[:set_title]("UTM Simulation")

        fig[:show]()

        vis.fig = fig
        vis.ax = ax
    else
        fig = vis.fig
        ax = vis.ax

        for artist in vis.artists
            artist[:set_visible](false)
        end
    end

    artists = PyObject[]
    

    if sc.cell_towers != nothing
        for (x, y) in sc.cell_towers
            cell_tower = ax[:plot](x, y, "k^", markerfacecolor = "white")
            append!(artists, cell_tower)
        end
    end


    if sc.landing_bases != nothing
        i = 1
        for (x, y) in sc.landing_bases
            landing_base = ax[:text](x, y, "H" * string(i), horizontalalignment = "center", verticalalignment = "center")
            push!(artists, landing_base)
            i += 1
        end
    end


    for uav in sc.UAVs
        planned_path = ax[:plot]([uav.start_loc[1]; map(x -> x[1], uav.waypoints); uav.end_loc[1]], [uav.start_loc[2]; map(x -> x[2], uav.waypoints); uav.end_loc[2]], ".--", color = "0.7")
        append!(artists, planned_path)
    end

    if vis.uav_start_locs == nothing
        vis.uav_start_locs = Vector{Float64}[]
        for uav_state in sc_state.UAVStates
            push!(vis.uav_start_locs, uav_state.curr_loc)
        end
    end

    for start_loc in vis.uav_start_locs
        uav_start_loc = ax[:plot](start_loc[1], start_loc[2], "k.")
        append!(artists, uav_start_loc)
    end


    fig[:canvas][:draw]()

    vis.artists = artists

    return vis
end


function visUpdate(vis::UTMVisualizer, sc::UTMScenario, sc_state::UTMScenarioState, timestep::Union{Int64, Void} = nothing, sim::Union{Tuple{ASCIIString, Vector{Float64}, Union{Int64, Float64}, Union{Int64, Float64}}, Void} = nothing)

    fig = vis.fig
    ax = vis.ax

    if timestep == nothing
        text = vis.ax[:text](0.5, -0.02, "$(round(Int64, sc.x))ft x $(round(Int64, sc.y))ft, scenario: $(sc.sn)", horizontalalignment = "center", verticalalignment = "top", transform = vis.ax[:transAxes])
    else
        if sim == nothing
            text = vis.ax[:text](0.5, -0.02, "timestep: $timestep, action: None_, observation: none, reward: 0, total reward: 0", horizontalalignment = "center", verticalalignment = "top", transform = vis.ax[:transAxes])
        else
            action, observation, r, R  = sim
            text = vis.ax[:text](0.5, -0.02, "timestep: $timestep, action: $action, observation: $(round(Int64, observation)), reward: $r, total reward: $R", horizontalalignment = "center", verticalalignment = "top", transform = vis.ax[:transAxes])
        end
    end
    push!(vis.artists, text)


    if timestep != nothing && timestep >= sc.jamming_time
        jamming_center_marker = ax[:plot](sc.jamming_center[1], sc.jamming_center[2], "kx", markersize = 5. / min(sc.x, sc.y) * 5280)
        append!(vis.artists, jamming_center_marker)

        jamming_area = ax[:add_patch](pathch.Circle((sc.jamming_center[1], sc.jamming_center[2]), radius = sc.jamming_radius, edgecolor = "0.5", facecolor = "none", linestyle = "dashed"))
        push!(vis.artists, jamming_area)
    end


    i = 1
    for uav_state in sc_state.UAVStates
        if uav_state.status == :flying
            path_alpha = 1.
            if i == 1
                marker_style = "bo"
            else
                marker_style = "mo"
            end
        else
            path_alpha = 0.2
            marker_style = "go"
        end

        if timestep != nothing
            uav_path = ax[:plot]([map(x -> x[1], uav_state.past_locs); uav_state.curr_loc[1]], [map(x -> x[2], uav_state.past_locs); uav_state.curr_loc[2]], "r", alpha = path_alpha)
            append!(vis.artists, uav_path)
        end

        if uav_state.status == :flying || uav_state.status == :landed
            uav_marker = ax[:plot](uav_state.curr_loc[1], uav_state.curr_loc[2], marker_style, markersize = 5. / min(sc.x, sc.y) * 5280)
            append!(vis.artists, uav_marker)

            if uav_state.curr_loc[1] >= 0 && uav_state.curr_loc[1] <= sc.x && uav_state.curr_loc[2] >= 0 && uav_state.curr_loc[2] <= sc.y
                uav_marker_text = ax[:text](uav_state.curr_loc[1] + 70, uav_state.curr_loc[2] + 70, string(i), size = 10, horizontalalignment = "center", verticalalignment = "center")
                push!(vis.artists, uav_marker_text)
            end
        end

        if timestep != nothing
            if uav_state.status == :flying
                uav_sa = ax[:add_patch](patch.Circle((uav_state.curr_loc[1], uav_state.curr_loc[2]), radius = sc.sa_dist / 2, edgecolor = "0.5", facecolor = "none", linestyle = "dotted"))
                push!(vis.artists, uav_sa)
            end
        end

        i += 1
    end


    fig[:canvas][:draw]()

    return vis
end


function updateAnimation(vis::UTMVisualizer, timestep::Union{Int64, Void} = nothing; bSaveFrame::Bool = false, filename::ASCIIString = "sim.png")

    append!(vis.ims, Any[vis.artists])

    if bSaveFrame
        if timestep == nothing
            savefig(filename, transparent = false)
        else
            base, ext = splitext(filename)
            savefig(base * "_" * string(timestep) * "." * ext, transparent = false)
        end
    end

    if vis.wait
        readline()
    end
end


function saveAnimation(vis::UTMVisualizer; interval::Int64 = 1000, repeat::Bool = false, filename::ASCIIString = "sim.mp4")

    if repeat || vis.wait
        readline()
        println("save animation")
    end

    im_ani = ani.ArtistAnimation(vis.fig, vis.ims, interval = interval, repeat = repeat, repeat_delay = interval * 5)
    im_ani[:save](filename)

    if repeat || vis.wait
        readline()
    end
end


end


