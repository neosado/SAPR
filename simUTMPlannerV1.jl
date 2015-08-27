# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 11/04/2014

using UTMPlannerV1_
using Scenario_
using UTMScenarioGenerator_

using POMCP_

using Util
using UTMVisualizer_
using MCTSVisualizer_

using Iterators
using Base.Test
using JLD


function sampleParticles(pm, b, nsample = 100000)

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
    sum_ = float(sum_)

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
        if s.Position == pm.rover_pos
            println(s, ": ", bv.belief[s])
        else
            @test bv.belief[s] == 0.
        end
    end
end


function getInitialBelief(pm::UTMPlannerV1)

    B = UPState[]

    push!(B, UPState(coord2grid(pm, pm.sc_state.UAVStates[1].curr_loc), pm.sc_state.UAVStates[1].status, pm.sc_state.UAVStates[1].heading, 0))

    return UPBeliefParticles(B)
end


function getInitialState(pm::UTMPlannerV1)

    return UPState(coord2grid(pm, pm.sc_state.UAVStates[1].curr_loc), pm.sc_state.UAVStates[1].status, pm.sc_state.UAVStates[1].heading, 0)
end


function test(pm, alg)

    b = getInitialBelief(pm)

    pm.sc.bMCTS = true
    a_opt, Q = selectAction(alg, pm, b)

    #println("T: ", alg.T)
    #println("N: ", alg.N)
    #println("Ns: ", alg.Ns)
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


function simulate(sc::Scenario, sc_state::ScenarioState; draw::Bool = false, wait::Bool = false)

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

function rollout_policy_(pm::UTMPlannerV1, s::UPState, param::Vector{Float64})

    while true
        a, prob = rand_action(pm, param)

        if isFeasible(pm, s, a) || a.action == s.heading
            return a
        end
    end
end

rollout_policy(param::Vector{Float64}) = (pm::UTMPlannerV1, s::UPState) -> rollout_policy_(pm, s, param)


function initRolloutPolicy(pm::UTMPlannerV1, alg::POMCP)

    dist_param = [0, ones(pm.nAction-1) / (pm.nAction-1)]

    alg.default_policy = rollout_policy(dist_param)

    return dist_param
end


function updateRolloutPolicy(pm::UTMPlannerV1, alg::POMCP, prev_dist_param::Vector{Float64}; gamma::Float64 = 1., rho::Float64 = 0.1)

    if alg.rollout_type == :CE_worst
        alpha = [0, ones(pm.nAction-1) * 0.05]

        nsample = length(alg.CE_samples)
        dist_param = zeros(pm.nAction)

        Z = zeros(nsample, pm.nAction)
        S = Array(Float64, nsample)
        W = Array(Float64, nsample)

        i = 1
        for (a, r) in alg.CE_samples
            a_ind = 0
            for j = 1:pm.nAction
                if a == pm.actions[j]
                    a_ind = j
                    break
                end
            end
            @assert a_ind != 0

            Z[i, a_ind] = 1
            S[i] = -r
            W[i] = (1 / pm.nAction) / prev_dist_param[a_ind]

            i += 1
        end

        Ssorted = sort(S)

        gamma_ = Ssorted[ceil((1 - rho) * nsample)]

        if gamma_ >= gamma
            gamma_ = gamma
        end

        I = map((x) -> x >= gamma_ ? 1 : 0, S)

        for i = 1:pm.nAction
            dist_param[i] = sum(I .* W .* Z[:, i]) / sum(I .* W) + alpha[i]
        end

        dist_param /= sum(dist_param)

    elseif alg.rollout_type == :CE_best
        if false
            alpha = [0, ones(pm.nAction-1) * 0.05]

            nsample = length(alg.CE_samples)
            dist_param = zeros(pm.nAction)

            Z = zeros(nsample, pm.nAction)
            S = Array(Float64, nsample)

            i = 1
            for (a, r) in alg.CE_samples
                a_ind = 0
                for j = 1:pm.nAction
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

            for i = 1:pm.nAction
                dist_param[i] = sum(I .* Z[:, i]) / sum(I) + alpha[i]
            end

            dist_param /= sum(dist_param)

        else
            alpha = [0, ones(pm.nAction-1) * 0.05]
            c = 1.

            dist_param = zeros(pm.nAction)

            R = Dict{UPAction, Vector{Float64}}()

            for a in pm.actions
                R[a] = Float64[]
            end

            for (a, r) in alg.CE_samples
                push!(R[a], r * pm.reward_norm_const)
            end

            for i = 1:pm.nAction
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

    alg.default_policy = rollout_policy(dist_param)

    return dist_param
end


function simulate(pm, alg; draw::Bool = false, wait::Bool = false, bSeq::Bool = false, ts::Int64 = 0, action::Symbol = :None_, bStat::Bool = false, debug::Int64 = 0)

    sc = pm.sc
    sc_state = pm.sc_state

    if draw
        upv= UTMVisualizer(wait = wait)
    end

    b = getInitialBelief(pm)

    s = getInitialState(pm)

    R = 0

    if alg != nothing && (alg.rollout_type == :CE_worst || alg.rollout_type == :CE_best)
        rollout_policy_param = initRolloutPolicy(pm, alg)
    end

    if draw
        visInit(upv, sc, sc_state)
        visUpdate(upv, sc, sc_state)
        updateAnimation(upv)

        visInit(upv, sc, sc_state)
        visUpdate(upv, sc, sc_state, s.t)
        updateAnimation(upv)
    end

    while s.t < length(pm.UAVStates)
        if bSeq
            #println("T: ", alg.T)
            #println("N: ", alg.N)
            #println("Ns: ", alg.Ns)
            #println("Q: ", alg.Q)
            #println("B: ", alg.B)
            #println()

            pm.sc.bMCTS = true
            if !bStat
                a, Q = selectAction(alg, pm, b, debug = debug)
            else
                a, Q, Qs = selectAction(alg, pm, b, bStat = true, debug = debug)

                for a__ in  pm.actions
                    data = Qs[a__]
                    println(a.action, ": ", neat(mean(data)), ", ", neat(std(data)), ", ", neat(std(data)/mean(data)))
                end
            end
            pm.sc.bMCTS = false

            #println("T: ", alg.T)
            #println("N: ", alg.N)
            #println("Ns: ", alg.Ns)
            #println("Q: ", alg.Q)
            #println("B: ", alg.B)
            #println()

            if alg.rollout_type == :CE_worst || alg.rollout_type == :CE_best
                if length(alg.CE_samples) != 0
                    rollout_policy_param = updateRolloutPolicy(pm, alg, rollout_policy_param)
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
                            println(a__, ": ", length(R__[a__]), ", ", neat(mean(R__[a__])))
                        end
                    end

                    println("ro_param: ", neat(rollout_policy_param))
                end
            end

        else
            a = UPAction(:None_)

            if s.t == ts
                a = UPAction(action)
            end

        end

        #s_ = nextState(pm, s, a)
        #o = observe(pm, s_, a)
        #r = reward(pm, s, a)

        s_, o, r = Generative(pm, s, a)

        R += r

        if debug > 0
            if debug > 2
                for a__ in pm.actions
                    println(a__, ": ", alg.N[(History(), a__)])
                end
            end

            Q__ = Float64[]
            if bSeq
                for a__ in  pm.actions
                    push!(Q__, Q[a__])
                end
            end

            println("time: ", s.t, ", s: ", grid2coord(pm, s.location), " ", s.status, ", Q: ", neat(Q__), ", a: ", a.action, ", o: ", grid2coord(pm, o.location), ", r: ", neat(r), ", R: ", neat(R), ", s_: ", grid2coord(pm, s_.location), " ", s_.status)
        end

        if draw
            visInit(upv, sc, sc_state)
            visUpdate(upv, sc, sc_state, s_.t, sim = (string(a.action), grid2coord(pm, o.location), r, R))
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

            # XXX add more particles
            if length(particles) == 0
                particles_ = UPState[]

                push!(particles_, UPState(o.location, s_.status, s_.heading, s_.t))

                loc = [o.location...]
                dist_min = Inf

                for h in keys(alg.B)
                    if length(h.history) == 2 && h.history[1] == a
                        loc_ = [h.history[2].location...]

                        if norm(loc_ - loc) < dist_min
                            dist_min = norm(loc_ - loc)
                        end
                    end
                end

                if dist_min * pm.cell_len < pm.sc.loc_err_bound
                    for h in sort(collect(keys(alg.B)), by = string)
                        if length(h.history) == 2 && h.history[1] == a
                            loc_ = [h.history[2].location...]

                            if norm(loc_ - loc) == dist_min
                                o = UPObservation(tuple(loc_...))
                                append!(particles_, getParticles(alg, a, o))
                            end
                        end
                    end
                end

                for s__ in particles_
                    if !isEnd(pm, s__)
                        push!(particles, s__)
                    end
                end
            end

            b = updateBelief(pm, UPBeliefParticles(particles))

            reinitialize(alg, a, o)
        end
    end

    if draw
        saveAnimation(upv, repeat = true)
    end

    return R
end


function default_policy(pm::UTMPlannerV1, s::UPState)

    a = pm.actions[rand(1:pm.nAction)]

    while !isFeasible(pm, s, a)
        a = pm.actions[rand(1:pm.nAction)]
    end

    return a
end


function evalScenario(scenario_number::Union(Int64, Nothing) = nothing; N::Int64 = 100, RE_threshold::Float64 = 0.1, bSeq::Bool = true, nloop::Int64 = 100, ts::Int64 = 0, action::Symbol = :None_, iseed::Union(Int64, Nothing) = nothing, rollout_type::Symbol = :default, Scenarios = nothing, debug::Int64 = 0)

    if iseed != nothing
        srand(iseed)
    end

    results = Float64[]
    va = Float64[]
    y = 0.

    n = 1

    while true
        if iseed != nothing
            pm = UTMPlannerV1(scenario_number = scenario_number, Scenarios = Scenarios)
        else
            seed = int64(time())

            if debug > 0
                print(seed, " ")
            end

            pm = UTMPlannerV1(seed = seed, scenario_number = scenario_number, Scenarios = Scenarios)
        end

        if !bSeq
            x = simulate(pm, nothing, ts = ts, action = action)
        else
            alg = POMCP(depth = 5, default_policy = default_policy, nloop_max = nloop, nloop_min = nloop, c = 2., gamma_ = 0.95, rollout_type = rollout_type, rgamma_ = 0.95)
            x = simulate(pm, alg, bSeq = bSeq)
        end

        push!(results, x)

        if debug > 0
            println(n, " ", x)
        end

        y += (x - y) / n
        push!(va, y)
        
        if n % 100 == 0
            if std(va) / abs(va[end]) < RE_threshold
                break
            end

            if debug > 0 && n != N
                println("n: ", n, ", mean: ", neat(va[end]), ", std: ", neat(std(va)), ", RE: ", neat(std(va) / abs(va[end])))
            end
        end

        if n == N
            break
        end

        n += 1
    end

    if debug > 0
        println("n: ", n, ", mean: ", neat(va[end]), ", std: ", neat(std(va)), ", RE: ", neat(std(va) / abs(va[end])))
    end

    return results
end


if false
    pm = UTMPlannerV1(seed = int64(time()))

    pm.sc.UAVs[1].navigation = :GPS_INS

    simulate(pm.sc, pm.sc_state, draw = true, wait = false)
end


if false
    srand(12)
    sn_list = unique(rand(1024:typemax(Int16), 1100))[1:10]

    println("scenarios: ", sn_list)
    #generateScenario(sn_list, bSave = true)

    Scenarios = loadScenarios()
    simulateScenario(sn_list, draw = true, wait = false, bSim = false, Scenarios = Scenarios)
end


if false
    simulateScenario(nothing, draw = true, wait = false, bSim = true)
end


if false
    seed = int64(time())
    #seed = 1440401261   # 100
    #seed = 1440609604   # 1000, CE_best, -2081
    #seed = 1440610899   # 1000, CE_best, -154

    scenario_number = nothing

    nloop = 100

    # :default, :default_once, :MC, :inf, :CE_worst, :CE_best
    rollout_type = :default

    pm = UTMPlannerV1(seed = seed, scenario_number = scenario_number)

    alg = POMCP(depth = 5, default_policy = default_policy, nloop_max = nloop, nloop_min = nloop, c = 2., gamma_ = 0.95, rollout_type = rollout_type, rgamma_ = 0.95, visualizer = MCTSVisualizer())

    #test(pm, alg)
    #simulate(pm, nothing, draw = true, wait = false, ts = 0, action = :None_)
    simulate(pm, alg, draw = true, wait = false, bSeq = true, bStat = false, debug = 1)
end


if false
    srand(12)
    sn_list = unique(rand(1024:typemax(Int16), 1100))[1:10]

    N = 100
    nloop = 100

    iseed = nothing

    debug = 0

    Scenarios = loadScenarios()

    for sn in sn_list
        println("scenario: ", sn)
        println("N: ", N, ", nloop: ", nloop, ", iseed: ", iseed)

        # :default, :default_once, :MC, :inf, :CE_worst, :CE_best
        for rollout_type in [:default, :CE_best, :CE_worst]
            print("rollout_type: ", rollout_type)
            if debug > 0
                println()
            end

            X = evalScenario(sn, N = N, nloop = nloop, iseed = iseed, rollout_type = rollout_type, Scenarios = Scenarios, debug = debug)

            if debug == 0
                print(", ")
            end
            println("mean: ", neat(mean(X)), ", std: ", neat(std(X)), ", RE: ", neat(std(X) / abs(mean(X))))
        end

        println()
    end
end


