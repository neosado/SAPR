# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 03/09/2015

module UTMVisualizer_

export UTMVisualizer, visInit, visUpdate, updateAnimation, saveAnimation


using Visualizer_
using Scenario_

using PyCall
using PyPlot

@pyimport matplotlib.animation as ani
@pyimport matplotlib.patches as patch

import Visualizer_.visInit
import Visualizer_.visUpdate
import Visualizer_.updateAnimation
import Visualizer_.saveAnimation


type UTMVisualizer <: Visualizer

    fig::Union(Figure, Nothing)
    ax1::Union(PyObject, Nothing)

    artists::Union(Vector{PyObject}, Nothing)

    ims::Vector{Any}

    wait::Bool

    uav_start_locs::Union(Vector{Vector{Float64}}, Nothing)


    function UTMVisualizer(;wait = false)

        self = new()

        self.fig = nothing
        self.ax1 = nothing

        self.artists = nothing

        self.ims = {}

        self.wait = wait

        self.uav_start_locs = nothing
        
        return self
    end
end


function visInit(vis::UTMVisualizer, sc::Scenario, sc_state::ScenarioState)

    if vis.fig == nothing
        fig = figure(facecolor = "white")

        ax1 = fig[:add_subplot](111)
        ax1[:set_aspect]("equal")
        ax1[:set_xlim](0, sc.x)
        ax1[:set_ylim](0, sc.y)
        ax1[:set_xticklabels]([])
        ax1[:set_yticklabels]([])
        ax1[:grid](true)
        ax1[:set_title]("UTM Simulation")

        fig[:show]()

        vis.fig = fig
        vis.ax1 = ax1
    else
        fig = vis.fig
        ax1 = vis.ax1

        for artist in vis.artists
            artist[:set_visible](false)
        end
    end

    artists = PyObject[]
    

    if sc.cell_towers != nothing
        for (x, y) in sc.cell_towers
            cell_tower = ax1[:plot](x, y, "k^", markerfacecolor = "white")
            append!(artists, cell_tower)
        end
    end


    if sc.landing_bases != nothing
        for (x, y) in sc.landing_bases
            landing_base = ax1[:text](x, y, "H", horizontalalignment = "center", verticalalignment = "center")
            push!(artists, landing_base)
        end
    end


    for uav in sc.UAVs
        planned_path = ax1[:plot]([uav.start_loc[1], map(x -> x[1], uav.waypoints), uav.end_loc[1]], [uav.start_loc[2], map(x -> x[2], uav.waypoints), uav.end_loc[2]], ".--", color = "0.7")
        append!(artists, planned_path)
    end

    if vis.uav_start_locs == nothing
        vis.uav_start_locs = Vector{Float64}[]
        for uav_state in sc_state.UAVStates
            push!(vis.uav_start_locs, uav_state.curr_loc)
        end
    end

    for start_loc in vis.uav_start_locs
        uav_start_loc = ax1[:plot](start_loc[1], start_loc[2], "k.")
        append!(artists, uav_start_loc)
    end


    fig[:canvas][:draw]()

    vis.artists = artists

    return vis
end


function visUpdate(vis::UTMVisualizer, sc::Scenario, sc_state::ScenarioState)

    fig = vis.fig
    ax1 = vis.ax1


    text = vis.ax1[:text](0.5, -0.02, "$(int(sc.x))ft x $(int(sc.y))ft, seed: $(sc.seed)", horizontalalignment = "center", verticalalignment = "top", transform = vis.ax1[:transAxes])
    push!(vis.artists, text)


    i = 1
    for uav_state in sc_state.UAVStates
        if i == 1
            marker_style = "bo"
        else
            marker_style = "mo"
        end

        uav_marker = ax1[:plot](uav_state.curr_loc[1], uav_state.curr_loc[2], marker_style, markersize = 5. / min(sc.x, sc.y) * 5280)
        append!(vis.artists, uav_marker)

        i += 1
    end


    fig[:canvas][:draw]()

    return vis
end


function visUpdate(vis::UTMVisualizer, sc::Scenario, sc_state::ScenarioState, timestep::Int64; sim::Union((ASCIIString, Union(Vector{Float64}, Nothing), Union(Int64, Float64), Union(Int64, Float64)), Nothing) = nothing)

    fig = vis.fig
    ax1 = vis.ax1


    if sim == nothing
        text = vis.ax1[:text](0.5, -0.02, "timestep: $timestep, action: None_, observation: none, reward: 0, total reward: 0", horizontalalignment = "center", verticalalignment = "top", transform = vis.ax1[:transAxes])
    else
        action, observation, r, R  = sim
        text = vis.ax1[:text](0.5, -0.02, "timestep: $timestep, action: $action, observation: $(int64(observation)), reward: $r, total reward: $R", horizontalalignment = "center", verticalalignment = "top", transform = vis.ax1[:transAxes])
    end
    push!(vis.artists, text)


    if timestep >= sc.jamming_time
        jamming_center_marker = ax1[:plot](sc.jamming_center[1], sc.jamming_center[2], "kx", markersize = 5. / min(sc.x, sc.y) * 5280)
        append!(vis.artists, jamming_center_marker)

        jamming_area = ax1[:add_patch](pathch.Circle((sc.jamming_center[1], sc.jamming_center[2]), radius = sc.jamming_radius, edgecolor = "0.5", facecolor = "none", linestyle = "dashed"))
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

        uav_path = ax1[:plot]([map(x -> x[1], uav_state.past_locs), uav_state.curr_loc[1]], [map(x -> x[2], uav_state.past_locs), uav_state.curr_loc[2]], "r", alpha = path_alpha)
        append!(vis.artists, uav_path)

        if uav_state.status == :flying || uav_state.status == :landed
            uav_marker = ax1[:plot](uav_state.curr_loc[1], uav_state.curr_loc[2], marker_style, markersize = 5. / min(sc.x, sc.y) * 5280)
            append!(vis.artists, uav_marker)
        end

        if uav_state.status == :flying
            uav_sa = ax1[:add_patch](patch.Circle((uav_state.curr_loc[1], uav_state.curr_loc[2]), radius = sc.sa_dist / 2, edgecolor = "0.5", facecolor = "none", linestyle = "dotted"))
            push!(vis.artists, uav_sa)
        end

        i += 1
    end


    fig[:canvas][:draw]()

    return vis
end


function updateAnimation(vis::UTMVisualizer, timestep::Int64 = -1; bSaveFrame::Bool = false, filename::ASCIIString = "sim.png")

    append!(vis.ims, {vis.artists})

    if bSaveFrame
        if timestep == -1
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


