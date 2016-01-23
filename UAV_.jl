# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 04/02/2015

VERSION >= v"0.4" && __precompile__(false)


module UAV_

export UAV, UAVState
export updateState, updateHeading, isEndState
export convertHeading


using Distributions


type UAV

    sc::Any

    index::Int64

    start_loc::Union{Vector{Float64}, Void}
    end_loc::Union{Vector{Float64}, Void}

    waypoints::Union{Vector{Vector{Float64}}, Void}
    nwaypoints::Int64

    velocity::Float64
    velocity_min::Float64
    velocity_max::Float64

    navigation::Symbol

    # for nav1
    IMU_gyr_sigma::Float64


    function UAV()

        self = new()

        self.sc = nothing

        self.index = 0

        self.start_loc = nothing    # [x:ft, y:ft]
        self.end_loc = nothing      # [x:ft, y:ft]

        self.waypoints = nothing    # list of [x:ft, y:ft]
        self.nwaypoints = 0

        self.velocity = 0.          # ft/s
        self.velocity_min = 0.
        self.velocity_max = 0.

        self.navigation = :GPS_INS

        self.IMU_gyr_sigma = 0.     # ft

        return self
    end
end


function convertHeading(uav::UAV, heading::Symbol)

    s = string(heading)

    if contains(s, "waypoint")
        htype = :waypoint
        hindex = parse(Int64, s[9:end])
        @assert hindex < uav.nwaypoints + 1

    elseif contains(s, "base")
        htype = :base
        hindex = parse(Int64, s[5:end])
        @assert hindex < length(uav.sc.landing_bases) + 1

    elseif contains(s, "end")
        htype = :end_
        hindex = 0

    else
        @assert false

    end

    if htype == :waypoint
        hloc = uav.waypoints[hindex]

    elseif htype == :base
        hloc = uav.sc.landing_bases[hindex]

    elseif htype == :end_
        hloc = uav.end_loc

    end

    return htype, hindex, hloc
end


type UAVState

    index::Int64

    status::Symbol

    curr_loc::Vector{Float64}
    past_locs::Vector{Vector{Float64}}

    heading::Symbol


    function UAVState(uav::UAV)

        self = new()

        self.index = uav.index

        self.status = :flying

        self.curr_loc = uav.start_loc
        self.past_locs = Vector{Float64}[]

        self.heading = :waypoint1

        return self
    end
end


function updateStateGPSINS(uav::UAV, state::UAVState)

    curr_loc = state.curr_loc

    htype, hindex, hloc = convertHeading(uav, state.heading)

    delta = (hloc - curr_loc) / norm(hloc - curr_loc) * uav.velocity * uav.sc.dt

    if norm(delta) > norm(hloc - curr_loc) || curr_loc == hloc
        dt = norm(hloc - curr_loc) / uav.velocity

        curr_loc = hloc

        if uav.sc.bMCTS == false
            push!(state.past_locs, curr_loc)
        end

        if htype == :waypoint
            if hindex == uav.nwaypoints
                state.heading = :end_
            else
                hindex += 1
                state.heading = symbol("waypoint" * string(hindex))
            end

            htype, hindex, hloc = convertHeading(uav, state.heading)

            curr_loc += (hloc - curr_loc) / norm(hloc - curr_loc) * uav.velocity * (uav.sc.dt - dt)
        end

    else
        curr_loc += delta

    end

    state.curr_loc = curr_loc
end


function updateStateNav1(uav::UAV, state::UAVState)

    curr_loc = state.curr_loc

    htype, hindex, hloc = convertHeading(uav, state.heading)

    if uav.sc.loc_err_sigma == 0
        loc_estimate = curr_loc
    else
        loc_estimate = rand(MvNormal(curr_loc, uav.sc.loc_err_sigma))
    end

    u = (hloc - loc_estimate) / norm(hloc - loc_estimate)
    angle_noise = randn() * uav.sc.heading_err_sigma
    u = [cosd(angle_noise) -sind(angle_noise); sind(angle_noise) cosd(angle_noise)] * u

    v = uav.velocity + randn() * uav.sc.velocity_err_sigma

    delta = u * v * uav.sc.dt

    if htype == :waypoint
        if hindex == uav.nwaypoints
            next_heading = :end_
        else
            hindex += 1
            next_heading = symbol("waypoint" * string(hindex))
        end

        next_htype, next_hindex, next_hloc = convertHeading(uav, next_heading)

        #if norm(hloc - loc_estimate) < uav.sc.loc_err_bound
        if norm(delta) > norm(hloc - loc_estimate) || dot(next_hloc - hloc, loc_estimate - hloc) > 0
            state.heading = next_heading
        end
    end

    state.curr_loc = curr_loc + delta
end


function updateState(uav::UAV, state::UAVState, t::Int64)

    if state.status == :flying
        if uav.sc.bMCTS == false
            push!(state.past_locs, state.curr_loc)
        end

        bInside = false

        if state.curr_loc[1] >= 0 && state.curr_loc[1] <= uav.sc.x && state.curr_loc[2] >= 0 && state.curr_loc[2] <= uav.sc.y
            bInside = true
        end

        if uav.navigation == :GPS_INS
            updateStateGPSINS(uav, state)

        elseif uav.navigation == :nav1
            updateStateNav1(uav, state)

        end

        curr_loc = state.curr_loc

        htype, hindex, hloc = convertHeading(uav, state.heading)

        if htype == :end_ || htype == :base
            if uav.navigation == :GPS_INS
                if curr_loc == hloc
                    state.status = :landed
                end

            elseif uav.navigation == :nav1
                if norm(curr_loc - hloc) < uav.sc.loc_err_bound
                    state.status = :landed
                end

            end
        end

        if curr_loc[1] < 0 || curr_loc[1] > uav.sc.x || curr_loc[2] < 0 || curr_loc[2] > uav.sc.y
            if bInside
                # assume that planned path does not come back in the area once it goes out of the area
                state.status = :out_of_area
            end
        end
    end
end


function updateHeading(uav::UAV, state::UAVState, heading::Symbol)

    state.heading = heading
end


function isEndState(uav::UAV, state::UAVState)

    if state.status == :landed || state.status == :out_of_area
        return true
    else
        return false
    end
end


end


