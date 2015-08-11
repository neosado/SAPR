# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 11/04/2014

using UTMPlannerV1_
using Scenario_

using POMCP_

using Util
using UTMVisualizer_
using MCTSVisualizer_

using Iterators
using Base.Test
using JSON


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

    push!(B, UPState(coord2grid(pm, pm.sc.UAVs[1].start_loc), :flying, :Waypoint1, 0))

    return UPBeliefParticles(B)
end


function getInitialState(pm::UTMPlannerV1)

    return UPState(coord2grid(pm, pm.sc.UAVs[1].start_loc), :flying, :Waypoint1, 0)
end


function test(pm, alg)

    b = getInitialBelief(pm)

    pm.sc.bMCTS = true
    a_opt, Qv = selectAction(alg, pm, b)

    #println("T: ", alg.T)
    #println("N: ", alg.N)
    #println("Ns: ", alg.Ns)
    #println("Q: ", alg.Q)
    #println("B: ", alg.B)
    #println()

    Qv__ = Float64[]
    for a in  pm.actions
        push!(Qv__, round(Qv[a], 2))
    end

    println("Qv: ", Qv__)
    println("action: ", a_opt.action)
end


function simulate(sc::Scenario, sc_state::ScenarioState; draw::Bool = false, wait::Bool = false, bDumpUAVStates::Bool = false)

    if draw
        vis = UTMVisualizer(wait = wait)

        visInit(vis, sc)
        visUpdate(vis, sc)
        updateAnimation(vis)
    end

    t = 0

    if bDumpUAVStates
        f = open("UTMPlannerV1.json", "w")
        S = Any[]
    end

    while !isEndState(sc, sc_state)
        updateState(sc, sc_state, t)

        if bDumpUAVStates
            push!(S, deepcopy(sc_state.UAVStates))
        end

        if draw
            visInit(vis, sc)
            visUpdate(vis, sc, sc_state, t)
            updateAnimation(vis)
        end

        t += 1
    end

    if bDumpUAVStates
        JSON.print(f, S)
        close(f)
    end

    if draw
        saveAnimation(vis, repeat = true)
    end
end


function simulate(pm, alg; draw::Bool = false, wait::Bool = false, bSeq::Bool = false)

    sc = pm.sc
    sc_state = pm.sc_state

    if draw
        upv= UTMVisualizer(wait = wait)
    end

    b = getInitialBelief(pm)

    s = getInitialState(pm)

    R = 0

    if draw
        visInit(upv, sc)
        visUpdate(upv, sc)
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
            a, Qv = selectAction(alg, pm, b)
            pm.sc.bMCTS = false

            #println("T: ", alg.T)
            #println("N: ", alg.N)
            #println("Ns: ", alg.Ns)
            #println("Q: ", alg.Q)
            #println("B: ", alg.B)
            #println()
        else
            a = UPAction(:None_)

            if s.t == 0
                a = UPAction(:None_)
            end

        end

        #s_ = nextState(pm, s, a)
        #o = observe(pm, s_, a)
        #r = reward(pm, s, a)

        s_, o, r = Generative(pm, s, a)

        R += r

        Qv__ = Float64[]
        if bSeq
            for a__ in  pm.actions
                push!(Qv__, round(Qv[a__], 2))
            end
        end

        if draw
            println("time: ", s.t, ", s: ", grid2coord(pm, s.location), " ", s.status, ", Qv: ", Qv__, ", a: ", a.action, ", o: ", grid2coord(pm, o.location), ", r: ", r, ", R: ", R, ", s_: ", grid2coord(pm, s_.location), " ", s_.status)

            visInit(upv, sc)
            visUpdate(upv, sc, sc_state, s.t, sim = (string(a.action), grid2coord(pm, o.location), r, R))
            updateAnimation(upv)
        end

        s = s_

        if isEnd(pm, s_)
            if draw
                println("reached the terminal state")

                visInit(upv, sc)
                visUpdate(upv, sc, sc_state, s_.t, sim = (string(a.action), grid2coord(pm, o.location), r, R))
                updateAnimation(upv)
            end

            break
        end

        if bSeq
            particles = getParticles(alg, a, o)

            # XXX add more particles
            if length(particles) == 0
                particles_ = UPState[]

                push!(particles_, UPState(o.location, s_.status, s_.heading, s_.t))

                loc = [o.location...]
                loc_min = loc
                dist_min = Inf

                for h in keys(alg.B)
                    if length(h.history) == 2 && h.history[1] == a
                        loc_ = [h.history[2].location...]

                        if norm(loc_ - loc) < dist_min
                            dist_min = norm(loc_ - loc)
                            loc_min = loc_
                        end
                    end
                end

                if dist_min * pm.cell_len < pm.sc.loc_err_bound
                    o = UPObservation(tuple(loc_min...))
                    append!(particles_, getParticles(alg, a, o))
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


if false
    pm = UTMPlannerV1(seed = int64(time()))

    pm.sc.UAVs[1].navigation = :GPS_INS
    pm.sc.sa_dist = 500.

    simulate(pm.sc, pm.sc_state, draw = true, wait = false, bDumpUAVStates = true)
end


if false
    pm = UTMPlannerV1(seed = int64(time()))

    #pm.sc.UAVs[1].navigation = :GPS_INS
    pm.sc.sa_dist = 500.

    alg = POMCP(depth = 5, default_policy = default_policy, nloop_max = 100, nloop_min = 100, c = 10., gamma_ = 0.95, rgamma_ = 0.95, visualizer = MCTSVisualizer())

    #test(pm, alg)
    simulate(pm, alg, draw = true, wait = false, bSeq = true)
end


if false
    N = 1000
    RE_threshold = 0.1
    bSeq = true

    va = Float64[]
    y = 0.

    n = 1
    while true
        pm = UTMPlannerV1(seed = int64(time()))

        #pm.sc.UAVs[1].navigation = :GPS_INS
        pm.sc.sa_dist = 500.

        # XXX debug
        print(pm.seed, " ")

        if !bSeq
            x = simulate(pm, nothing)
        else
            alg = POMCP(depth = 5, default_policy = default_policy, nloop_max = 100, nloop_min = 100, c = 10., gamma_ = 0.95, rgamma_ = 0.95)
            x = simulate(pm, alg, bSeq = bSeq)
        end

        # XXX debug
        println(n, " ", x)

        y += (x - y) / n
        push!(va, y)
        
        if n % 100 == 0
            if std(va) / abs(va[end]) < RE_threshold
                break
            end
        end

        if n == N
            break
        end

        n += 1
    end

    println("n: ", n, ", mean: ", va[end], ", std: ", std(va), ", RE: ", std(va) / abs(va[end]))
end


