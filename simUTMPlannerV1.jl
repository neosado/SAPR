# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 11/04/2014

push!(LOAD_PATH, ".")

using UTMPlannerV1_
using UTMScenario_
using UTMScenarioGenerator_

using POMCP_

using UTMVisualizer_
using MCTSVisualizer_

using ArmRewardModel_
using Util

using CEOpt_

using Iterators
using Distributions
using Base.Test
using JLD

import UAV_.convertHeading


function sampleParticles(pm, b, nsample = 1000)

    B = UPState[]

    for n = 1:nsample
        rv = rand()

        sum_ = 0.
        for s in keys(b.belief)
            sum_ += b.belief[s]

            if rv < sum_
                push!(B, s)

                break
            end
        end
    end

    return UPBeliefParticles(B)
end


function beliefParticles2Vector(pm, B)

    count_ = Dict{UPState, Int64}()
    belief = Dict{UPState, Float64}()

    for s in pm.states
        count_[s] = 0
        belief[s] = 0.
    end

    sum_ = 0
    for s in B.particles
        count_[s] += 1
        sum_ += 1
    end
    sum_ = Float64(sum_)

    for s in B.particles
        belief[s] = count_[s] / sum_
    end

    return UPBeliefVector(belief)
end


function printBelief(pm, alg, b)

    if typeof(alg) == POMCP
        bv = beliefParticles2Vector(pm, b)
    else
        bv = b
    end

    for s in pm.states
        println(s, ": ", bv.belief[s])
    end
end


function getInitialBelief(pm::UTMPlannerV1)

    B = UPState[]

    htype, hindex, hloc = convertHeading(pm.sc.UAVs[1], pm.sc_state.UAVStates[1].heading)

    if htype == :waypoint
        if hindex == 1
            last_ = :none
        else
            last_ = symbol("waypoint" * string(hindex - 1))
        end
    elseif htype == :end_
        last_ = symbol("waypoint" * string(pm.sc.UAVs[1].nwaypoints))
    end

    push!(B, UPState(coord2grid(pm, pm.sc_state.UAVStates[1].curr_loc), pm.sc_state.UAVStates[1].status, last_, 0))

    return UPBeliefParticles(B)
end


function getInitialState(pm::UTMPlannerV1)

    htype, hindex, hloc = convertHeading(pm.sc.UAVs[1], pm.sc_state.UAVStates[1].heading)

    if htype == :waypoint
        if hindex == 1
            last_ = :none
        else
            last_ = symbol("waypoint" * string(hindex - 1))
        end
    elseif htype == :end_
        last_ = symbol("waypoint" * string(pm.sc.UAVs[1].nwaypoints))
    end

    return UPState(coord2grid(pm, pm.sc_state.UAVStates[1].curr_loc), pm.sc_state.UAVStates[1].status, last_, 0)
end


function test(pm, alg)

    b = getInitialBelief(pm)

    pm.sc.bMCTS = true
    a_opt, Q = selectAction(alg, pm, b)

    #println("T: ", alg.T)
    #println("Ns: ", alg.Ns)
    #println("N: ", alg.N)
    #println("Q: ", alg.Q)
    #println("B: ", alg.B)
    #println()

    Q__ = Float64[]
    for a in  pm.actions
        push!(Q__, Q[a])
    end

    println("Q: ", neat(Q__))
    println("action: ", a_opt.action)
end


function simulate(sc::UTMScenario, sc_state::UTMScenarioState; draw::Bool = false, wait::Bool = false)

    if draw
        vis = UTMVisualizer(wait = wait)

        visInit(vis, sc, sc_state)
        visUpdate(vis, sc, sc_state)
        updateAnimation(vis)
    end

    t = 0

    while !isEndState(sc, sc_state)
        updateState(sc, sc_state, t)

        if draw
            visInit(vis, sc, sc_state)
            visUpdate(vis, sc, sc_state, t)
            updateAnimation(vis)
        end

        t += 1
    end

    if draw
        saveAnimation(vis, repeat = true)
    end
end


function default_policy(pm::UTMPlannerV1, s::UPState)
    
    # Note: pass pm.rng to rand() if pm supports rng

    a = pm.actions[rand(1:pm.nActions)]

    while !isFeasible(pm, s, a)
        a = pm.actions[rand(1:pm.nActions)]
    end

    return a
end


function rollout_default(alg::POMCP, pm::UTMPlannerV1, s::UPState, h::History, d::Int64; rgamma::Float64 = 0.9, debug::Int64 = 0)

    if d == 0 || isEnd(pm, s) || h.history == []
        return 0
    end

    a = default_policy(pm, s)

    if debug > 3
        print(string(a), ", ")
    end

    s_, o, r = alg.Generative(pm, s, a)

    return r + rgamma * rollout_default(alg, pm, s_, History([h.history; a; o]), d - 1, rgamma = rgamma, debug = debug)
end


function rollout_MC(alg::POMCP, pm::UTMPlannerV1, s::UPState, h::History, d::Int64; n::Int64 = 10, rgamma::Float64 = 0.9, debug::Int64 = 0)

    q = 0.

    for i = 1:n
        q += (rollout_default(alg, pm, s, h, d, rgamma = rgamma) - q) / i
    end

    return q
end


function rollout_none(alg::POMCP, pm::UTMPlannerV1, s::UPState, h::History, d::Int64; rgamma::Float64 = 0.9, debug::Int64 = 0)

    if d == 0 || isEnd(pm, s) || h.history == []
        return 0
    end

    a = h.history[end-1]

    s_, o, r = alg.Generative(pm, s, a)

    return r + rgamma * rollout_none(alg, pm, s_, History([h.history; a; o]), d - 1, rgamma = rgamma)
end


function rollout_refined(alg::POMCP, pm::UTMPlannerV1, s::UPState, h::History, d::Int64; rgamma::Float64 = 0.9, debug::Int64 = 0)

    if h.history == []
        return 0

    elseif isEnd(pm, s)
        return 0

    elseif d == 0
        r = 0

        uav = pm.sc.UAVs[1]

        atype, aindex, aloc = convertHeading(uav, h.history[end-1].action)

        if atype == :base
            r += -round(Int64, ceil(norm(aloc - grid2coord(pm, s.location)) / uav.velocity))

        elseif atype == :waypoint
            r += -round(Int64, ceil(norm(aloc - grid2coord(pm, s.location)) / uav.velocity))
            for i = aindex:uav.nwaypoints-1
                r += -round(Int64, ceil(norm(uav.waypoints[i+1] - uav.waypoints[i]) / uav.velocity))
            end
            r += -round(Int64, ceil(norm(uav.end_loc - uav.waypoints[uav.nwaypoints]) / uav.velocity))
            r += (uav.nwaypoints - aindex + 1) * 100 + 100

        elseif atype == :end_
            r += -round(Int64, ceil(norm(uav.end_loc - grid2coord(pm, s.location)) / uav.velocity))
            r += 100

        end

        return r

    end

    a = h.history[end-1]

    s_, o, r = alg.Generative(pm, s, a)

    return r + rgamma * rollout_refined(alg, pm, s_, History([h.history; a; o]), d - 1, rgamma = rgamma)
end


function rollout_inf(alg::POMCP, pm::UTMPlannerV1, s::UPState, h::History, d::Int64; rgamma::Float64 = 0.9, debug::Int64 = 0)

    q = 0.

    a = h.history[end-1]

    i = 0
    while !isEnd(pm, s)
        s_, o, r = alg.Generative(pm, s, a)
        q += rgamma^i * r
        i += 1
        s = s_
    end

    return q
end


function rollout_once(alg::POMCP, pm::UTMPlannerV1, s::UPState, h::History, d::Int64; rgamma::Float64 = 0.9, debug::Int64 = 0)

    if d == 0 || isEnd(pm, s)
        return 0
    end

    a = default_policy(pm, s)

    s_, o, r = alg.Generative(pm, s, a)

    return r + rgamma * rollout_none(alg, pm, s_, History([h.history; a; o]), d - 1, rgamma = rgamma)
end


function rand_action(pm::UTMPlannerV1, param::Vector{Float64})

    a = nothing
    prob = nothing

    param_cumulated = cumsum(param)

    X = rand()

    for i = 1:length(param_cumulated)
        if X <= param_cumulated[i]
            a = pm.actions[i]
            prob = param[i]
            break
        end
    end

    return a, prob
end

function CE_rollout_policy_(pm::UTMPlannerV1, s::UPState, param::Vector{Float64})

    while true
        a, prob = rand_action(pm, param)

        if isFeasible(pm, s, a)
            return a
        end
    end
end

CE_rollout_policy(param::Vector{Float64}) = (pm::UTMPlannerV1, s::UPState) -> CE_rollout_policy_(pm, s, param)


function rollout_CE(alg::POMCP, pm::UTMPlannerV1, s::UPState, h::History, d::Int64; rgamma::Float64 = 0.9, debug::Int64 = 0)

    if d == 0
        return 0
    end

    a = CE_rollout_policy(alg.CE_dist_param)(pm, s)

    if debug > 2
        print(string(a), ", ")
    end

    @assert isFeasible(pm, s, a)

    s_, o, r = alg.Generative(pm, s, a)

    if isEnd(pm, s_)
        push!(alg.CE_samples, (a, r))
        return r
    end

    q = r + rgamma * rollout_none(alg, pm, s_, History([h.history; a; o]), d - 1, rgamma = rgamma)

    push!(alg.CE_samples, (a, r))

    return q
end


function initRolloutPolicyForCE(pm::UTMPlannerV1, alg::POMCP)

    dist_param = [0, ones(pm.nActions-1) / (pm.nActions-1)]

    alg.CE_dist_param = dist_param

    return dist_param
end


function updateRolloutPolicyForCE(pm::UTMPlannerV1, alg::POMCP, prev_dist_param::Vector{Float64}; gamma::Float64 = 1., rho::Float64 = 0.1)

    if alg.rollout_type == :CE_worst
        alpha = [0, ones(pm.nActions-1) * 0.05]

        nsample = length(alg.CE_samples)
        dist_param = zeros(pm.nActions)

        Z = zeros(nsample, pm.nActions)
        S = Array(Float64, nsample)
        W = Array(Float64, nsample)

        i = 1
        for (a, r) in alg.CE_samples
            a_ind = 0
            for j = 1:pm.nActions
                if a == pm.actions[j]
                    a_ind = j
                    break
                end
            end
            @assert a_ind != 0

            Z[i, a_ind] = 1
            S[i] = -r
            W[i] = (1 / pm.nActions) / prev_dist_param[a_ind]

            i += 1
        end

        Ssorted = sort(S)

        gamma_ = Ssorted[ceil((1 - rho) * nsample)]

        if gamma_ >= gamma
            gamma_ = gamma
        end

        I = map((x) -> x >= gamma_ ? 1 : 0, S)

        for i = 1:pm.nActions
            dist_param[i] = sum(I .* W .* Z[:, i]) / sum(I .* W) + alpha[i]
        end

        dist_param /= sum(dist_param)

    elseif alg.rollout_type == :CE_best
        if false
            alpha = [0, ones(pm.nActions-1) * 0.05]

            nsample = length(alg.CE_samples)
            dist_param = zeros(pm.nActions)

            Z = zeros(nsample, pm.nActions)
            S = Array(Float64, nsample)

            i = 1
            for (a, r) in alg.CE_samples
                a_ind = 0
                for j = 1:pm.nActions
                    if a == pm.actions[j]
                        a_ind = j
                        break
                    end
                end
                @assert a_ind != 0

                Z[i, a_ind] = 1
                S[i] = r

                i += 1
            end

            Ssorted = sort(S)

            gamma_ = Ssorted[ceil((1 - rho) * nsample)]

            I = map((x) -> x >= gamma_ ? 1 : 0, S)

            for i = 1:pm.nActions
                dist_param[i] = sum(I .* Z[:, i]) / sum(I) + alpha[i]
            end

            dist_param /= sum(dist_param)

        else
            alpha = [0, ones(pm.nActions-1) * 0.05]
            c = 1.

            dist_param = zeros(pm.nActions)

            R = Dict{UPAction, Vector{Float64}}()

            for a in pm.actions
                R[a] = Float64[]
            end

            for (a, r) in alg.CE_samples
                push!(R[a], r * pm.reward_norm_const)
            end

            for i = 1:pm.nActions
                dist_param[i] = mean(R[pm.actions[i]])
            end

            dist_param = 1 ./ (-dist_param + c)

            indexes = find(x -> !isequal(x, NaN), dist_param)
            indexes_ = find(x -> isequal(x, NaN), dist_param)

            dist_param[indexes] /= sum(dist_param[indexes])
            dist_param[indexes] += alpha[indexes]
            dist_param[indexes_] = alpha[indexes_]

            dist_param /= sum(dist_param)

        end

    end

    alg.CE_dist_param = dist_param

    return dist_param
end


function rollout_MS(alg::POMCP, pm::UTMPlannerV1, s::UPState, h::History, d::Int64; MSState::Union{Dict{ASCIIString, Any}, Void} = nothing, rgamma::Float64 = 0.9, debug::Int64 = 0)

    @test MSState != nothing

    if d == 0 || isEnd(pm, s) || h.history == []
        return 0
    end

    n = 1

    loc = [pm.cell_len / 2 + (s.location[1] - 1) * pm.cell_len, pm.cell_len / 2 + (s.location[2] - 1) * pm.cell_len]

    for i = 2:pm.sc.nUAV
        # XXX replicate once when hitting a level first time
        if MSState["level"][i] < length(alg.tree_policy.ms_L) + 1
            loc_ = pm.sc_state.UAVStates[i].curr_loc

            if norm(loc - loc_) < alg.tree_policy.ms_L[MSState["level"][i]]
                if debug > 2
                    println("    UAV 1 ", loc, " and UAV ", i, " ", neat(loc_), " hit the MS level ", MSState["level"][i], " at ro level ", d)
                end

                if alg.tree_policy.ms_N[MSState["level"][i]] > n
                    n = alg.tree_policy.ms_N[MSState["level"][i]]
                end

                MSState["level"][i] += 1
            end
        end
    end

    #a = h.history[end-1]
    a = default_policy(pm, s)

    q_ = 0.

    for k = 1:n
        s_, o, r = alg.Generative(pm, s, a)

        q = r + rgamma * rollout_MS(alg, pm, s_, History([h.history; a; o]), d - 1, MSState = deepcopy(MSState), rgamma = rgamma, debug = debug)

        q_ += (q - q_) / k
    end

    return q_
end


function rollout_MS_refined(alg::POMCP, pm::UTMPlannerV1, s::UPState, h::History, d::Int64; MSState::Union{Dict{ASCIIString, Any}, Void} = nothing, rgamma::Float64 = 0.9, debug::Int64 = 0)

    @test MSState != nothing

    if h.history == []
        return 0

    elseif isEnd(pm, s)
        return 0

    elseif d == 0
        r = 0

        uav = pm.sc.UAVs[1]

        atype, aindex, aloc = convertHeading(uav, h.history[end-1].action)

        if atype == :base
            r += -round(Int64, ceil(norm(aloc - grid2coord(pm, s.location)) / uav.velocity))

        elseif atype == :waypoint
            r += -round(Int64, ceil(norm(aloc - grid2coord(pm, s.location)) / uav.velocity))
            for i = aindex:uav.nwaypoints-1
                r += -round(Int64, ceil(norm(uav.waypoints[i+1] - uav.waypoints[i]) / uav.velocity))
            end
            r += -round(Int64, ceil(norm(uav.end_loc - uav.waypoints[uav.nwaypoints]) / uav.velocity))
            r += (uav.nwaypoints - aindex + 1) * 100 + 100

        elseif atype == :end_
            r += -round(Int64, ceil(norm(uav.end_loc - grid2coord(pm, s.location)) / uav.velocity))
            r += 100

        end

        return r

    end

    n = 1

    loc = [pm.cell_len / 2 + (s.location[1] - 1) * pm.cell_len, pm.cell_len / 2 + (s.location[2] - 1) * pm.cell_len]

    for i = 2:pm.sc.nUAV
        # XXX replicate once when hitting a level first time
        if MSState["level"][i] < length(alg.tree_policy.ms_L) + 1
            loc_ = pm.sc_state.UAVStates[i].curr_loc

            if norm(loc - loc_) < alg.tree_policy.ms_L[MSState["level"][i]]
                if debug > 2
                    println("    UAV 1 ", loc, " and UAV ", i, " ", neat(loc_), " hit the MS level ", MSState["level"][i], " at level ", d)
                end

                if alg.tree_policy.ms_N[MSState["level"][i]] > n
                    n = alg.tree_policy.ms_N[MSState["level"][i]]
                end

                MSState["level"][i] += 1
            end
        end
    end

    a = h.history[end-1]

    q_ = 0.

    for k = 1:n
        s_, o, r = alg.Generative(pm, s, a)

        q = r + rgamma * rollout_MS(alg, pm, s_, History([h.history; a; o]), d - 1, MSState = deepcopy(MSState), rgamma = rgamma, debug = debug)

        q_ += (q - q_) / k
    end

    return q_
end


function simulate(pm, alg; draw::Bool = false, wait::Bool = false, bSeq::Bool = true, ts::Int64 = 0, action::Symbol = :waypoint1, bStat::Bool = false, debug::Int64 = 0)

    sc = pm.sc
    sc_state = pm.sc_state

    if draw
        upv= UTMVisualizer(wait = wait)
    end

    b = getInitialBelief(pm)

    s = getInitialState(pm)

    R = 0

    if alg != nothing && (alg.rollout_type == :CE_worst || alg.rollout_type == :CE_best)
        CE_rollout_policy_param = initRolloutPolicyForCE(pm, alg)
    end

    if draw
        visInit(upv, sc, sc_state)
        visUpdate(upv, sc, sc_state)
        updateAnimation(upv)

        visInit(upv, sc, sc_state)
        visUpdate(upv, sc, sc_state, s.t)
        updateAnimation(upv)
    end

    #while s.t < length(pm.UAVStates)
    while true
        if bSeq
            #println("T: ", alg.T)
            #println("Ns: ", alg.Ns)
            #println("N: ", alg.N)
            #println("Q: ", alg.Q)
            #println("B: ", alg.B)
            #println()

            if alg.rollout_type == :CE_worst || alg.rollout_type == :CE_best
                alg.CE_samples = Tuple{Action, Int64}[]
            end

            pm.sc.bMCTS = true
            if !bStat
                a, Q = selectAction(alg, pm, b, debug = debug)
            else
                a, Q, Qs = selectAction(alg, pm, b, bStat = true, debug = debug)

                for a__ in  pm.actions
                    data = Qs[a__]
                    println(string(a), ": ", neat(mean(data)), ", ", neat(std(data)), ", ", neat(std(data)/mean(data)))
                end
            end
            pm.sc.bMCTS = false

            #println("T: ", alg.T)
            #println("Ns: ", alg.Ns)
            #println("N: ", alg.N)
            #println("Q: ", alg.Q)
            #println("B: ", alg.B)
            #println()

            Q_max = -Inf
            for a__ in pm.actions
                if !isFeasible(pm, s, a__)
                    Q[a__] = -Inf
                end

                if Q[a__] > Q_max
                    Q_max = Q[a__]
                end
            end

            actions = UPAction[]
            for a__ in pm.actions
                if Q[a__] == Q_max
                    push!(actions, a__)
                end
            end
            a = actions[rand(1:length(actions))]

            if alg.rollout_type == :CE_worst || alg.rollout_type == :CE_best
                if length(alg.CE_samples) != 0
                    CE_rollout_policy_param = updateRolloutPolicyForCE(pm, alg, CE_rollout_policy_param)
                end

                if debug > 1
                    if debug > 2
                        R__ = Dict{UPAction, Vector{Float64}}()
                        for a__ in pm.actions
                            R__[a__] = Float64[]
                        end
                        for (a__, r__) in alg.CE_samples
                            push!(R__[a__], r__ * pm.reward_norm_const)
                        end
                        for a__ in pm.actions
                            println(string(a__), ": ", length(R__[a__]), ", ", neat(mean(R__[a__])))
                        end
                    end

                    println("CE_ro_param: ", neat(CE_rollout_policy_param))
                end
            end

        else
            a = UPAction(:waypoint1)

            if s.t == ts
                a = UPAction(action)
            end

        end

        #s_ = nextState(pm, s, a)
        #o = observe(pm, s_, a)
        #r = reward(pm, s, a, s_)

        s_, o, r = Generative(pm, s, a)

        R += r

        if debug > 0
            if debug > 3
                for a__ in pm.actions
                    println(string(a__), ": ", alg.N[(History(), a__)])
                end
            end

            Q__ = Float64[]
            if bSeq
                for a__ in  pm.actions
                    push!(Q__, Q[a__])
                end
            end

            println("ts: ", s.t, ", s: ", grid2coord(pm, s.location), " ", s.status, " ", s.last, ", Q: ", neat(Q__), ", a: ", string(a), ", o: ", grid2coord(pm, o.location), ", r: ", neat(r), ", R: ", neat(R), ", s_: ", grid2coord(pm, s_.location), " ", s_.status, " ", s_.last)
        end

        if draw
            visInit(upv, sc, sc_state)
            visUpdate(upv, sc, sc_state, s_.t, (string(a), grid2coord(pm, o.location), neat(r), neat(R)))
            updateAnimation(upv)
        end

        s = s_

        if isEnd(pm, s_)
            if debug > 0
                println("reached the terminal state")
            end

            break
        end

        if bSeq
            particles = getParticles(alg, a, o)

            if length(particles) != 0
                particles_ = UPState[]

                for s__ in particles
                    if !isEnd(pm, s__)
                        push!(particles_, s__)
                    end
                end

                particles = particles_
            end

            if length(particles) == 0
                push!(particles, UPState(o.location, s_.status, s_.last, s_.t))
            end

            # add particles
            if length(particles) < 10
                particles_ = UPState[]

                for i = 1:10
                    s__ = particles[rand(1:length(particles))]
                    push!(particles_, UPState(coord2grid(pm, rand(MvNormal(grid2coord(pm, s__.location), pm.sc.loc_err_sigma))), s__.status, s__.last, s__.t))
                end

                append!(particles, particles_)
            end

            b = updateBelief(pm, UPBeliefParticles(particles))

            min_dist = Inf
            min_obs = nothing
            for o__ in alg.Os[(History(), a)]
                dist = norm([o__.location[1] - o.location[1], o__.location[2] - o.location[2]])
                if dist < min_dist
                    min_dist = dist
                    min_obs = o__
                end
            end
            o = min_obs

            reinitialize(alg, a, o)
        end
    end

    if draw
        saveAnimation(upv, repeat = true)
    end

    return R
end


function evalScenario(scenario::Int64, N::Int64, up_seed::Union{Int64, Vector{Int64}}, mcts_seed::Union{Int64, Vector{Int64}}, tree_policy::Any, rollout::Union{Tuple{Symbol, Function}, Void}; depth::Int64 = 10, nloop_min::Int64 = 100, nloop_max::Int64 = 1000, runtime_max::Float64 = 0., bSeq::Bool = true, ts::Int64 = 0, action::Symbol = :waypoint1, RE_threshold::Float64 = 0., Scenarios::Any = nothing, debug::Int64 = 0)

    @assert length(up_seed) == N
    @assert length(mcts_seed) == N

    X = Float64[]

    meanX = 0.
    ssX = 0.
    varX = 0.
    RE = 0.

    n = 0

    for i = 1:N
        n = i

        if debug > 0
            print(i, " ", up_seed[i], " ", mcts_seed[i], " ")
        end

        pm = UTMPlannerV1(seed = up_seed[i], scenario = scenario, Scenarios = Scenarios)

        if !bSeq
            x = simulate(pm, nothing, bSeq = false, ts = ts, action = action)
        else
            alg = POMCP(seed = mcts_seed[i], depth = depth, nloop_min = nloop_min, nloop_max = nloop_max, runtime_max = runtime_max, gamma_ = 0.9, tree_policy = tree_policy, rollout = rollout)
            x = simulate(pm, alg)
        end

        push!(X, x)

        if debug > 0
            println(x)
        end

        meanX += (x - meanX) / i
        ssX += x * x

        if i > 1
            varX = (ssX - i * (meanX * meanX)) / ((i - 1) * i)
            RE = sqrt(varX) / abs(meanX)

            if RE < RE_threshold
                break
            end
        end

        if i % 10 == 0
            if debug > 0 && i != N
                println("n: ", i, ", mean: ", neat(meanX), ", std: ", neat(sqrt(varX)), ", RE: ", neat(RE))
            end
        end
    end

    if debug > 1
        println(X)
    end

    if debug > 0
        println("n: ", n, ", mean: ", neat(meanX), ", std: ", neat(sqrt(varX)), ", RE: ", neat(RE))
    end

    return X
end


function runExp(scenario::Int64, up_seed::Union{Int64, Vector{Int64}}, mcts_seed::Union{Int64, Vector{Int64}}, tree_policy::Any, N::Int64; depth::Int64 = 5, nloop_min::Int64 = 100, nloop_max::Int64 = 10000, runtime_max::Float64 = 1., bParallel::Bool = false, id::Any = nothing)

    @assert length(up_seed) == N
    @assert length(mcts_seed) == N

    bMS = false

    if tree_policy != nothing
        for stp in tree_policy
            if stp["type"] == :MS
                bMS = true
            end
        end
    end

    if bMS
        #rollout = (:MS, rollout_MS_refined)
        rollout = (:MS, rollout_MS)
    else
        #rollout = (:refined, rollout_refined)
        rollout = nothing
    end

    opt_return = nothing
    returns = zeros(N)

    for i = 1:N
        pm = UTMPlannerV1(seed = up_seed[i], scenario = scenario)

        if i == 1
            opt_return = (pm.sc.UAVs[1].nwaypoints + 1) * 100
        end

        alg = POMCP(seed = mcts_seed[i], depth = depth, nloop_min = nloop_min, nloop_max = nloop_max, runtime_max = runtime_max, gamma_ = 0.9, tree_policy = tree_policy, rollout = rollout)

        R = simulate(pm, alg)

        returns[i] = R
    end

    if N == 1
        returns = returns[1]
    end

    if bParallel
        return id, opt_return, returns
    else
        return opt_return, returns
    end
end


function drawSample(p)

    if p[2] == 0.
        return p[1]
    else
        return rand(Truncated(Normal(p[1], p[2]), 0, Inf))
    end
end

function computePerf_(scenario::Int64, tree_policy::Any, depth::Int64, nloop_min::Int64, nloop_max::Int64, runtime_max::Float64, rollout::Union{Tuple{Symbol, Function}, Void}, id, x)

    for stp in tree_policy
        if stp["type"] == :UCB1
            stp["c"] = x
        end
    end

    pm = UTMPlannerV1(scenario = scenario)

    alg = POMCP(depth = depth, nloop_min = nloop_min, nloop_max = nloop_max, runtime_max = runtime_max, gamma_ = 0.9, tree_policy = tree_policy, rollout = rollout)

    R = simulate(pm, alg)

    if id == nothing
        return R
    else
        return id, R
    end
end

computePerf(scenario, tree_policy, depth, nloop_min, nloop_max, runtime_max, rollout) = (id, x) -> computePerf_(scenario, tree_policy, depth, nloop_min, nloop_max, runtime_max, rollout, id, x)

function updateParam(X, S, gamma_)

    I = map((x) -> x >= gamma_ ? 1 : 0, S)

    p = Array(Float64, 2)
    p[1] = sum(I .* X) / sum(I)
    p[2]= sqrt(sum(I .* (X - p[1]).^2) / sum(I))

    return p
end


function expBatchWorker(scenarios::Union{Int64, Vector{Int64}}, tree_policies, depth::Int64, nloop_min::Int64, nloop_max::Int64, runtime_max::Float64, N::Int64; bParallel::Bool = false, datafile::ASCIIString = "exp.jld", bAppend::Bool = false)

    if !bAppend && isfile(datafile)
        rm(datafile)
    end

    for scenario in scenarios
        println("Scenario: ", scenario)

        srand(scenario)

        up_seed_list = unique(rand(10000:typemax(Int16), round(Int64, N * 1.1)))[1:N]
        mcts_seed_list = unique(rand(10000:typemax(Int16), round(Int64, N * 1.1)))[1:N]

        R = Dict{Tuple{Int64, ASCIIString}, Dict{ASCIIString, Any}}()

        if bParallel
            if true
                for tree_policy in tree_policies
                    bMS = false
                    bUCB1withCE = false

                    if tree_policy != nothing
                        for stp in tree_policy
                            if stp["type"] == :MS
                                bMS = true
                            elseif stp["type"] == :UCB1withCE
                                bUCB1withCE = true
                            end
                        end
                    end

                    if bUCB1withCE
                        tree_policy_ = deepcopy(tree_policy)

                        for stp in tree_policy_
                            if stp["type"] == :UCB1withCE
                                stp["type"] = :UCB1
                                stp["c"] = 1.
                            end
                        end

                        if bMS
                            rollout = (:MS, rollout_MS)
                        else
                            rollout = nothing
                        end

                        p = CEOpt(drawSample, [100, 1000], computePerf(scenario, tree_policy_, depth, nloop_min, nloop_max, runtime_max, rollout), updateParam, 100, 0.0460517, bParallel = true)

                        for stp in tree_policy_
                            if stp["type"] == :UCB1
                                stp["c"] = p[1]
                            end
                        end

                    else
                        tree_policy_ = tree_policy

                    end

                    results = pmap(id -> runExp(scenario, up_seed_list[id], mcts_seed_list[id], tree_policy_, 1, depth = depth, nloop_min = nloop_min, nloop_max = nloop_max, runtime_max = runtime_max, bParallel = true, id = id), 1:N)

                    opt_return = 0
                    returns = zeros(N)

                    for result in results
                        id = result[1]
                        opt_return = result[2]
                        returns[id] = result[3]
                    end

                    R[(scenario, string(tree_policy))] = Dict("up_seed_list" => copy(up_seed_list), "mcts_seed_list" => copy(mcts_seed_list), "N" => N, "depth" => depth, "nloop_min" => nloop_min, "nloop_max" => nloop_max, "runtime_max" => runtime_max, "opt_return" => opt_return, "returns" => returns)
                end

            else
                results = pmap(tree_policy -> runExp(scenario, up_seed_list, mcts_seed_list, tree_policy, N, depth = depth, nloop_min = nloop_min, nloop_max = nloop_max, runtime_max = runtime_max, bParallel = true, id = tree_policy), tree_policies)

                for result in results
                    tree_policy = result[1]
                    opt_return = result[2]
                    returns = result[3]

                    R[(scenario, string(tree_policy))] = Dict("up_seed_list" => copy(up_seed_list), "mcts_seed_list" => copy(mcts_seed_list), "N" => N, "depth" => depth, "nloop_min" => nloop_min, "nloop_max" => nloop_max, "runtime_max" => runtime_max, "opt_return" => opt_return, "returns" => returns)
                end

            end

        else
            for tree_policy in tree_policies
                opt_return, returns = runExp(scenario, up_seed_list, mcts_seed_list, tree_policy, N, depth = depth, nloop_min = nloop_min, nloop_max = nloop_max, runtime_max = runtime_max)
                R[(scenario, string(tree_policy))] = Dict("up_seed_list" => copy(up_seed_list), "mcts_seed_list" => copy(mcts_seed_list), "N" => N, "depth" => depth, "nloop_min" => nloop_min, "nloop_max" => nloop_max, "runtime_max" => runtime_max, "opt_return" => opt_return, "returns" => returns)
            end

        end

        if isfile(datafile)
            D = load(datafile)

            Scenarios = D["Scenarios"]
            TreePolicies = D["TreePolicies"]
            Results = D["Results"]

            for (key, experiment) in R
                scenario, tree_policy = key

                if !(scenario in Scenarios)
                    push!(Scenarios, scenario)
                    TreePolicies[scenario] = map(string, tree_policies)
                elseif !(tree_policy in TreePolicies[scenario])
                    push!(TreePolicies[scenario], tree_policy)
                end

                Results[(scenario, tree_policy)] = experiment
            end

        else
            Scenarios = Int64[scenario]

            TreePolicies = Dict{Int64, Vector{ASCIIString}}()
            TreePolicies[scenario] = map(string, tree_policies)

            Results = R

        end

        save(datafile, "Scenarios", Scenarios, "TreePolicies", TreePolicies, "Results", Results)
    end
end


function runExpBatch(; bParallel::Bool = false, bAppend::Bool = false)

    srand(12)
    nScenarios = 100

    scenarios = unique(rand(10000:typemax(Int16), round(Int64, nScenarios * 1.1)))[1:nScenarios]

    sparse_ = Dict("type" => :SparseUCT, "nObsMax" => 4)
    MS = Dict("type" => :MS, "L" => [500., 200.], "N" => [2, 2])

    tree_policies = Any[
        Any[sparse_, Dict("type" => :UCB1, "c" => 100)],
        Any[sparse_, Dict("type" => :UCB1, "c" => 10000)],
        Any[sparse_, Dict("type" => :TS)],
        Any[sparse_, Dict("type" => :TSM, "ARM" => () -> ArmRewardModel(0.01, 0.01, -100., 1., 1 / 2, 1 / (2 * (1 / 10. ^ 2)), -5000., -10000., 1., 1 / 2,  1 / (2 * (1 / 1.^2))))],
        Any[sparse_, Dict("type" => :AUCB, "SP" => [Dict("type" => :UCB1, "c" => 100), Dict("type" => :UCB1, "c" => 10000)])],
        Any[sparse_, Dict{ASCIIString, Any}("type" => :UCB1withCE)]
    ]

    depth = 10

    nloop_min = 1000
    nloop_max = 1000
    runtime_max = 0.

    N = 100

    datafile = "exp.jld"

    expBatchWorker(scenarios, tree_policies, depth, nloop_min, nloop_max, runtime_max, N, bParallel = bParallel, datafile = datafile, bAppend = bAppend)
end


if false
    pm = UTMPlannerV1(seed = round(Int64, time()))

    pm.sc.UAVs[1].navigation = :GPS_INS

    simulate(pm.sc, pm.sc_state, draw = true, wait = false)
end


if false
    srand(12)
    scenarios = unique(rand(1025:typemax(Int16), 1100))[1:10]

    println("scenarios: ", scenarios)
    #generateScenario(scenarios, bSave = true)

    Scenarios = loadScenarios()
    simulateScenario(scenarios, draw = true, wait = false, bSim = false, Scenarios = Scenarios)
end


if false
    #simulateScenario()
    simulateScenario(nothing, draw = true, wait = false, bSim = true)
    #simulateScenario(1, draw = true, wait = false, bSim = true, navigation = :nav1)
    #for i = 1:10
    #    simulateScenario(nothing, draw = true, wait = false, bSim = true)
    #end
end


if false
    srand(12)
    nScenarios = 100
    scenarios = unique(rand(10000:typemax(Int16), round(Int64, nScenarios * 1.1)))[1:nScenarios]
    for i = 1:nScenarios
        simulateScenario(scenarios[i], draw = true, wait = false, bSim = false)
    end
end


if false
    #scenario = rand(1025:typemax(Int16))
    scenario = 1

    up_seed = round(Int64, time())
    mcts_seed = round(Int64, time()) + 1

    println("scenario: ", scenario, ", seed: ", up_seed, ", ", mcts_seed)

    depth = 10

    nloop_min = 1000
    nloop_max = 1000
    runtime_max = 0.

    sparse_ = Dict("type" => :SparseUCT, "nObsMax" => 4)
    pw = Dict("type" => :ProgressiveWidening, "c" => 2, "alpha" => 0.4)
    MS = Dict("type" => :MS, "L" => [500., 200.], "N" => [2, 2])

    #tree_policy = nothing
    #tree_policy = Any[sparse_, Dict("type" => :UCB1, "c" => 100)]
    tree_policy = Any[sparse_, Dict("type" => :UCB1, "c" => 10000)]
    #tree_policy = Any[sparse_, Dict("type" => :TS)]
    #tree_policy = Any[sparse_, Dict("type" => :TSM, "ARM" => () -> ArmRewardModel(0.01, 0.01, -100., 1., 1 / 2, 1 / (2 * (1 / 10. ^ 2)), -5000., -10000., 1., 1 / 2,  1 / (2 * (1 / 1.^2))))]
    #tree_policy = Any[sparse_, Dict("type" => :AUCB, "SP" => [Dict("type" => :UCB1, "c" => 100), Dict("type" => :UCB1, "c" => 10000)])]

    # :default, :MC, :inf, :once, :CE_worst, :CE_best, :MS
    rollout = nothing
    #rollout = (:refined, rollout_refined)
    #rollout = (:MS, rollout_MS)
    #rollout = (:MS, rollout_MS_refined)

    debug = 2


    pm = UTMPlannerV1(seed = up_seed, scenario = scenario)

    alg = POMCP(seed = mcts_seed, depth = depth, nloop_min = nloop_min, nloop_max = nloop_max, runtime_max = runtime_max, gamma_ = 0.9, tree_policy = tree_policy, rollout = rollout, visualizer = MCTSVisualizer())

    #test(pm, alg)
    #simulate(pm, nothing, draw = true, wait = false, ts = 0, action = :waypoint1)
    simulate(pm, alg, draw = true, wait = false, bSeq = true, bStat = false, debug = debug)
end


if false
    srand(12)
    nScenarios = 100

    scenarios = unique(rand(10000:typemax(Int16), round(Int64, nScenarios * 1.1)))[1:nScenarios]

    sparse_ = Dict("type" => :SparseUCT, "nObsMax" => 4)
    MS = Dict("type" => :MS, "L" => [500., 200.], "N" => [2, 2])

    tree_policies = Any[
        Any[sparse_, Dict("type" => :UCB1, "c" => 100)],
        Any[sparse_, Dict("type" => :UCB1, "c" => 10000)],
        Any[sparse_, Dict("type" => :TS)],
        Any[sparse_, Dict("type" => :TSM, "ARM" => () -> ArmRewardModel(0.01, 0.01, -100., 1., 1 / 2, 1 / (2 * (1 / 10. ^ 2)), -5000., -10000., 1., 1 / 2,  1 / (2 * (1 / 1.^2))))],
        Any[sparse_, Dict("type" => :AUCB, "SP" => [Dict("type" => :UCB1, "c" => 100), Dict("type" => :UCB1, "c" => 10000)])]
    ]

    #tree_policies = Any[
    #    nothing,
    #    Dict("type" => :SparseUCT, "nObsMax" => 1),
    #    Dict("type" => :SparseUCT, "nObsMax" => 2),
    #    Dict("type" => :SparseUCT, "nObsMax" => 4),
    #    Dict("type" => :SparseUCT, "nObsMax" => 6),
    #    Dict("type" => :SparseUCT, "nObsMax" => 8),
    #    Dict("type" => :SparseUCT, "nObsMax" => 10),
    #    Dict("type" => :SparseUCT, "nObsMax" => 12),
    #    Dict("type" => :ProgressiveWidening, "c" => 1, "alpha" => 0.4),
    #    Dict("type" => :ProgressiveWidening, "c" => 2, "alpha" => 0.4),
    #    Dict("type" => :ProgressiveWidening, "c" => 4, "alpha" => 0.4),
    #    Dict("type" => :ProgressiveWidening, "c" => 6, "alpha" => 0.4),
    #    Dict("type" => :ProgressiveWidening, "c" => 2, "alpha" => 0.1),
    #    Dict("type" => :ProgressiveWidening, "c" => 2, "alpha" => 0.2),
    #    Dict("type" => :ProgressiveWidening, "c" => 2, "alpha" => 0.4),
    #    Dict("type" => :ProgressiveWidening, "c" => 2, "alpha" => 0.6)
    #]

    # :default, :MC, :inf, :once, :CE_worst, :CE_best, :MS
    rollouts = [(:default, rollout_default)]
    #rollouts = [(:refined, rollout_refined)]
    #rollouts = [(:MS, rollout_MS)]
    #rollouts = [(:MS, rollout_MS_refined)]

    depth = 10

    nloop_min = 1000
    nloop_max = 1000
    runtime_max = 0.

    N = 100
    RE_threshold = 0.

    debug = 0


    Scenarios = loadScenarios()

    for scenario in scenarios
        println("scenario: ", scenario)
        println("N: ", N, ", RE_threshold: ", RE_threshold, ", depth: ", depth, ", nloop_min: ", nloop_min, ", nloop_max: ", nloop_max, ", runtime_max: ", runtime_max)

        srand(scenario)

        up_seed_list = unique(rand(10000:typemax(Int16), round(Int64, N * 1.1)))[1:N]
        mcts_seed_list = unique(rand(10000:typemax(Int16), round(Int64, N * 1.1)))[1:N]

        for tree_policy in tree_policies
            for rollout in rollouts
                println("tree_policy: ", tree_policy, ", rollout: ", rollout)

                X = evalScenario(scenario, N, up_seed_list, mcts_seed_list, tree_policy, rollout, depth = depth, nloop_min = nloop_min, nloop_max = nloop_max, runtime_max = runtime_max, RE_threshold = RE_threshold, Scenarios = Scenarios, debug = debug)

                println("n: ", length(X), ", mean: ", neat(mean(X)), ", std: ", neat(std(X) / sqrt(length(X))), ", RE: ", neat((std(X ) / sqrt(length(X))) / abs(mean(X))))
            end
        end

        println()
    end
end


