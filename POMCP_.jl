# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 11/04/2014

# Partially Observable Monte-Carlo Planning
# D. Silver and J. Veness, "Monte-Carlo Planning in Large POMDPs," in Advances in Neural Information Processing Systems (NIPS), 2010.

module POMCP_

export POMCP, selectAction, reinitialize, initialize, getParticles


using MCTS_
using POMDP_
using Util
using MCTSVisualizer_

using Base.Test


import MCTS_.selectAction
import MCTS_.reinitialize
import MCTS_.initialize


type POMCP <: MCTS

    seed::Union(Int64, Nothing)

    depth::Int64

    default_policy::Function

    T::Dict{History, Bool}
    N::Dict{(History, Action), Int64}
    Ns::Dict{History, Int64}
    Q::Dict{(History, Action), Float64}
    B::Dict{History, Vector{State}}

    nloop_max::Int64
    nloop_min::Int64
    eps::Float64

    c::Float64
    gamma_::Float64

    rollout_type::Symbol
    rgamma_::Float64
    CE_samples::Vector{(Action, Float64)}

    reuse::Bool

    visualizer::Union(MCTSVisualizer, Nothing)


    function POMCP(;seed::Union(Int64, Nothing) = nothing, depth::Int64 = 3, default_policy::Function = pi_0, nloop_max::Int64 = 10000, nloop_min::Int64 = 10000, eps::Float64 = 1.e-3, c::Float64 = 1., gamma_::Float64 = 0.9, rollout_type::Symbol = :default, rgamma_::Float64 = 0.9, visualizer::Union(MCTSVisualizer, Nothing) = nothing)

        self = new()

        if seed != nothing
            if seed != 0
                self.seed = seed
            else
                self.seed = int64(time())
            end

            srand(self.seed)
        end

        self.depth = depth

        self.default_policy = default_policy

        self.T = Dict{History, Bool}()
        self.N = Dict{(History, Action), Int64}()
        self.Ns = Dict{History, Int64}()
        self.Q = Dict{(History, Action), Float64}()

        self.B = Dict{History, Vector{State}}()

        self.nloop_max = nloop_max
        self.nloop_min = nloop_min
        self.eps = eps

        self.c = c
        self.gamma_ = gamma_

        self.rollout_type = rollout_type
        self.rgamma_ = rgamma_  # rollout gamma
        self.CE_samples = (Action, Int64)[]

        self.reuse = false

        self.visualizer = visualizer

        return self
    end
end


# \pi_0
function pi_0(pm::POMDP, s::State)
    
    a = pm.actions[rand(1:pm.nAction)]

    while !isFeasible(pm, s, a)
        a = pm.actions[rand(1:pm.nAction)]
    end

    return a
end


function Generative(pm::POMDP, s::State, a::Action)

    return POMDP_.Generative(pm, s, a)
end


function rollout_default(alg::POMCP, pm::POMDP, s::State, h::History, d::Int64)

    if d == 0
        return 0
    end

    a = alg.default_policy(pm, s)
    @assert isFeasible(pm, s, a)

    s_, o, r = Generative(pm, s, a)

    if isEnd(pm, s_)
        return r
    end

    return r + alg.rgamma_ * rollout_default(alg, pm, s_, h, d - 1)
end


function rollout_inf(alg::POMCP, pm::POMDP, s::State, h::History, d::Int64)

    R = 0.

    while true
        s_, o, r = Generative(pm, s, pm.actions[1])
        R += r
        s = s_

        if isEnd(pm, s)
            break
        end
    end

    return R
end


function rollout_none(alg::POMCP, pm::POMDP, s::State, h::History, d::Int64)

    if d == 0
        return 0
    end

    s_, o, r = Generative(pm, s, pm.actions[1])

    if isEnd(pm, s_)
        return r
    end

    return r + alg.rgamma_ * rollout_none(alg, pm, s_, h, d - 1)
end


function rollout_default_once(alg::POMCP, pm::POMDP, s::State, h::History, d::Int64)

    if d == 0
        return 0
    end

    a = alg.default_policy(pm, s)
    @assert isFeasible(pm, s, a)

    s_, o, r = Generative(pm, s, a)

    if isEnd(pm, s_)
        return r
    end

    return r + alg.rgamma_ * rollout_none(alg, pm, s_, h, d - 1)
end


function rollout_CE(alg::POMCP, pm::POMDP, s::State, h::History, d::Int64)

    if d == 0
        return 0
    end

    a = alg.default_policy(pm, s)
    @assert a.action != :None_

    if a.action == s.heading
        a_ = pm.actions[1]
    else
        a_ = a
    end

    @assert isFeasible(pm, s, a_)

    s_, o, r = Generative(pm, s, a_)

    if isEnd(pm, s_)
        push!(alg.CE_samples, (a, r))
        return r
    end

    r += alg.rgamma_ * rollout_none(alg, pm, s_, h, d - 1)
    push!(alg.CE_samples, (a, r))

    return r
end


function rollout(alg::POMCP, pm::POMDP, s::State, h::History, d::Int64)

    if alg.rollout_type == :default
        r = rollout_default(alg, pm, s, h, d)

    elseif alg.rollout_type == :default_once
        r = rollout_default_once(alg, pm, s, h, d)

    elseif alg.rollout_type == :MC
        r = 0

        for n = 1:10
            r += (rollout_default(alg, pm, s, h, d + 3) - r) / n
        end

    elseif alg.rollout_type == :inf
        r = rollout_inf(alg, pm, s, h, d)

    elseif alg.rollout_type == :CE_worst || alg.rollout_type == :CE_best
        r = rollout_CE(alg, pm, s, h, d)

    end

    return r
end


function simulate(alg::POMCP, pm::POMDP, s::State, h::History, d::Int64; bStat::Bool = false)

    if alg.visualizer != nothing
        updateTree(alg.visualizer, :start_sim, s)
    end

    if d == 0 || isEnd(pm, s)
        if !haskey(alg.T, h)
            for a in pm.actions
                alg.N[(h, a)] = 0

                if isFeasible(pm, s, a)
                    alg.Q[(h, a)] = 0
                else
                    alg.Q[(h, a)] = -Inf
                end
            end

            alg.Ns[h] = 1
            alg.T[h] = true
            alg.B[h] = [s]

        else
            push!(alg.B[h], s)

        end

        return 0
    end

    if !haskey(alg.T, h)
        #println("new node: ", h, " at level ", d)

        for a in pm.actions
            alg.N[(h, a)] = 0

            if isFeasible(pm, s, a)
                alg.Q[(h, a)] = 0
            else
                alg.Q[(h, a)] = -Inf
            end
        end

        alg.Ns[h] = 1
        alg.T[h] = true
        alg.B[h] = [s]

        ro = rollout(alg, pm, s, h, d)
        #println("rollout: ", ro)

        return ro
    end

    #println("found node: ", h, " at level ", d)

    Qv = Array(Float64, pm.nAction)
    for i = 1:pm.nAction
        a = pm.actions[i]

        if !isFeasible(pm, s, a)
            Qv[i] = -Inf
        elseif alg.N[(h, a)] == 0
            Qv[i] = Inf
        else
            Qv[i] = alg.Q[(h, a)] + alg.c * sqrt(log(alg.Ns[h]) / alg.N[(h, a)])
        end
    end

    a = pm.actions[argmax(Qv)]

    s_, o, r = Generative(pm, s, a)

    #println("Qv: ", round(Qv, 2), ", (a, o): {", a, ", ", o, "}, s_: ", s_, ", r: ", r)

    if alg.visualizer != nothing
        updateTree(alg.visualizer, :before_sim, s, a, o)
    end

    q = r + alg.gamma_ * simulate(alg, pm, s_, History([h.history, a, o]), d - 1)

    alg.N[(h, a)] += 1
    alg.Ns[h] += 1
    alg.Q[(h, a)] += (q - alg.Q[(h, a)]) / alg.N[(h, a)]

    push!(alg.B[h], s)

    if alg.visualizer != nothing
        updateTree(alg.visualizer, :after_sim, s, a, r, q, alg.N[(h, a)], alg.Ns[h], alg.Q[(h, a)])
    end

    if bStat && d == alg.depth
        return q, a
    else
        return q
    end
end


function selectAction(alg::POMCP, pm::POMDP, b::Belief; bStat::Bool = false)

    if alg.visualizer != nothing
        initTree(alg.visualizer)
    end

    h = History()

    if !alg.reuse
        initialize(alg)
    end

    Qv = Dict{Action, Float64}()
    for a in pm.actions
        Qv[a] = 0.
    end

    if bStat
        Qv_data = Dict{Action, Vector{Float64}}()

        for a in pm.actions
            Qv_data[a] = Float64[]
        end
    end

    if alg.rollout_type == :CE_worst || alg.rollout_type == :CE_best
        alg.CE_samples = (Action, Int64)[]
    end

    for i = 1:alg.nloop_max
        #println("iteration: ", i)

        s = sampleBelief(pm, b)

        if !bStat
            simulate(alg, pm, s, h, alg.depth)
        else
            ret = simulate(alg, pm, s, h, alg.depth, bStat = true)
            if length(ret) == 2
                q, a = ret
                push!(Qv_data[a], q)
            end
        end

        #println("h: ", h)
        #println("T: ", alg.T)
        #println("N: ", alg.N)
        #println("Ns: ", alg.Ns)
        #println("Q: ", alg.Q)
        #println()

        res = 0.

        for a in pm.actions
            Qv_prev = Qv[a]
            Qv[a] = alg.Q[(h, a)]
            res += (Qv[a] - Qv_prev)^2
        end

        if i > alg.nloop_min &&  sqrt(res) < alg.eps
            break
        end
    end

    Qv_max = -Inf
    for a in pm.actions
        Qv[a] = alg.Q[(h, a)]

        if Qv[a] > Qv_max
            Qv_max = Qv[a]
        end
    end

    actions = Action[]
    for a in pm.actions
        if Qv[a] == Qv_max
            push!(actions, a)
        end
    end
    action = actions[rand(1:length(actions))]

    if alg.visualizer != nothing
        saveTree(alg.visualizer, pm)
    end

    if !bStat
        return action, Qv
    else
        return action, Qv, Qv_data
    end
end


function reinitialize(alg::POMCP, a::Action, o::Observation)

    T_new = Dict{History, Bool}()
    N_new = Dict{(History, Action), Int64}()
    Ns_new = Dict{History, Int64}()
    Q_new = Dict{(History, Action), Float64}()
    B_new = Dict{History, Vector{State}}()

    for h in keys(alg.T)
        if length(h.history) > 0 && h.history[1] == a && h.history[2] == o
            T_new[History(h.history[3:end])] = alg.T[h]
            Ns_new[History(h.history[3:end])] = alg.Ns[h]
        end
    end

    for h in keys(alg.B)
        if length(h.history) > 0 && h.history[1] == a && h.history[2] == o
            B_new[History(h.history[3:end])] = alg.B[h]
        end
    end

    for key in keys(alg.N)
        h, action = key
        if length(h.history) > 0 && h.history[1] == a && h.history[2] == o
            N_new[(History(h.history[3:end]), action)] = alg.N[key]
            Q_new[(History(h.history[3:end]), action)] = alg.Q[key]
        end
    end

    alg.T = T_new
    alg.N = N_new
    alg.Ns = Ns_new
    alg.Q = Q_new
    alg.B = B_new

    alg.reuse = true

    if alg.visualizer != nothing
        alg.visualizer.b_hist_acc = true
    end
end


function initialize(alg::POMCP)

    alg.T = Dict{History, Bool}()
    alg.N = Dict{(History, Action), Int64}()
    alg.Ns = Dict{History, Int64}()
    alg.Q = Dict{(History, Action), Float64}()
    alg.B = Dict{History, Vector{State}}()

    alg.reuse = false

    if alg.visualizer != nothing
        alg.visualizer.b_hist_acc = false
    end
end


function getParticles(alg::POMCP, a::Action, o::Observation)

    if haskey(alg.B, History([a, o]))
        return alg.B[History([a, o])]
    else
        return State[]
    end
end


end


