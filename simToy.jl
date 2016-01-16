# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 08/25/2015

using Toy_

using POMCP_

using Util
using MCTSVisualizer_


function simulate(pm, alg; debug::Int64 = 0)

    b = TYBeliefParticles([TYState(:S1)])

    s = TYState(:S1)

    R = 0.

    n = 0

    while !isEnd(pm, s)
        a, Q = selectAction(alg, pm, b, debug = debug)

        s_, o, r = Generative(pm, s, a)

        n += 1

        R += r

        if debug > 0
            if debug > 2
                for a__ in pm.actions
                    println(a__, ": ", alg.N[(History(), a__)])
                end
            end

            Q__ = Float64[]
            for a__ in  pm.actions
                push!(Q__, Q[a__])
            end

            println("step: ", n, ", s: ", s.state, ", Q: ", neat(Q__), ", a: ", a.action, ", o: ", o.observation, ", r: ", r, ", R: ", R, ", s_: ", s_.state)
        end

        s = s_

        #b = updateBelief(pm, TYBeliefParticles(getParticles(alg, a, o)))
        b = TYBeliefParticles([TYState(symbol(string(o.observation)[1:2]))])

        reinitialize(alg, a, o)
    end

    return R
end


function rand_action(pm::Toy, param::Vector{Float64})

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

function rollout_policy_(pm::Toy, s::TYState, param::Vector{Float64})

    while true
        a, prob = rand_action(pm, param)

        if isFeasible(pm, s, a)
            return a
        end
    end
end

rollout_policy(param::Vector{Float64}) = (pm::Toy, s::TYState) -> rollout_policy_(pm, s, param)


if false
    pm = Toy(seed = round(Int64, time()))

    n = 0

    while !isEnd(pm, s)
        a = pm.actions[rand(1:2)]

        s_, o, r = Generative(pm, s, a)

        n += 1

        R += r

        println("step: ", n, ", s: ", s.state, ", a: ", a.action, ", o: ", o.observation, ", r: ", r, ", R: ", R, ", s_: ", s_.state)

        s = s_
    end
end


if false
    seed = round(Int64, time())

    p = 0.5

    nloop = 10

    println("seed: ", seed)
    println("p: ", p)

    pm = Toy(seed = seed)

    alg = POMCP(depth = 2, default_policy = rollout_policy([p, 1 - p]), nloop_max = nloop, nloop_min = nloop, c = 2., gamma_ = 0.95, rgamma_ = 0.95, visualizer = MCTSVisualizer())

    simulate(pm, alg, debug = 3)
end


function simTest(;N = 1000, RE_threshold = 0.1, p = 0.5, nloop = 100)

    iseed = round(Int64, time())

    if iseed != nothing
        println("seed: ", iseed)
        srand(iseed)
    end

    println("p: ", p)

    va = Float64[]
    y = 0.

    n = 1
    while true
        if iseed != nothing
            pm = Toy()
        else
            seed = round(Int64, time())
            pm = Toy(seed = seed)
        end

        if iseed == nothing
            print(pm.seed, " ")
        end

        alg = POMCP(depth = 2, default_policy = rollout_policy([p, 1 - p]), nloop_max = nloop, nloop_min = nloop, c = 2., gamma_ = 0.95, rgamma_ = 0.95, visualizer = MCTSVisualizer())

        x = simulate(pm, alg)

        #println(n, " ", x)

        y += (x - y) / n
        push!(va, y)
        
        if n % 100 == 0
            if std(va) / abs(va[end]) < RE_threshold
                break
            end

            if n != N
                #println("step: ", n, ", mean: ", neat(va[end]), ", std: ", neat(std(va)), ", RE: ", neat(std(va) / abs(va[end])))
            end
        end

        if n == N
            break
        end

        n += 1
    end

    println("step: ", n, ", mean: ", neat(va[end]), ", std: ", neat(std(va)), ", RE: ", neat(std(va) / abs(va[end])))
end


if false
    for p = 0.:0.1:1.
        simTest(N = 100, RE_threshold = 0.1, p = p, nloop = 10)
    end
end



