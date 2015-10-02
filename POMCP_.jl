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

    Generative::Function

    default_policy::Function

    T::Dict{History, Bool}
    N::Dict{(History, Action), Int64}
    Ns::Dict{History, Int64}
    Q::Dict{(History, Action), Float64}

    X2::Dict{(History, Action), Float64}

    B::Dict{History, Vector{State}}
    Os::Dict{(History, Action), Vector{Observation}}

    nloop_max::Int64
    nloop_min::Int64
    eps::Float64

    runtime_max::Float64

    c::Float64
    gamma_::Float64

    rollout_type::Symbol
    rollout_func::Function
    rgamma_::Float64
    CE_samples::Vector{(Action, Float64)}

    reuse::Bool

    visualizer::Union(MCTSVisualizer, Nothing)


    function POMCP(;seed::Union(Int64, Nothing) = nothing, depth::Int64 = 3, default_policy::Function = pi_0, nloop_max::Int64 = 10000, nloop_min::Int64 = 10000, eps::Float64 = 1.e-3, runtime_max::Float64 = 0., c::Float64 = 1., gamma_::Float64 = 0.9, rollout::Union((Symbol, Function), Nothing) = nothing, rgamma_::Float64 = 0.9, visualizer::Union(MCTSVisualizer, Nothing) = nothing)

        self = new()

        if seed != nothing
            if seed != 0
                self.seed = seed
            else
                self.seed = int64(time())
            end

            srand(self.seed)

        else
            self.seed = nothing

        end

        self.depth = depth

        self.Generative = Generative

        self.default_policy = default_policy

        self.T = Dict{History, Bool}()
        self.N = Dict{(History, Action), Int64}()
        self.Ns = Dict{History, Int64}()
        self.Q = Dict{(History, Action), Float64}()

        self.X2 = Dict{(History, Action), Float64}()

        self.B = Dict{History, Vector{State}}()
        self.Os = Dict{(History, Action), Vector{Observation}}()

        self.nloop_max = nloop_max
        self.nloop_min = nloop_min
        self.eps = eps

        self.runtime_max = runtime_max

        self.c = c
        self.gamma_ = gamma_

        if rollout == nothing
            self.rollout_type = :default
            self.rollout_func = rollout_default
        else
            self.rollout_type = rollout[1]
            self.rollout_func = rollout[2]
        end

        self.rgamma_ = rgamma_  # rollout gamma
        self.CE_samples = (Action, Int64)[]

        self.reuse = false

        self.visualizer = visualizer

        return self
    end
end


function Generative(pm::POMDP, s::State, a::Action)

    s_, o, r = POMDP_.Generative(pm, s, a)

    return s_, o, r / pm.reward_norm_const
end


# \pi_0
function pi_0(pm::POMDP, s::State)
    
    a = pm.actions[rand(1:pm.nAction)]

    while !isFeasible(pm, s, a)
        a = pm.actions[rand(1:pm.nAction)]
    end

    return a
end


function rollout_default(alg::POMCP, pm::POMDP, s::State, h::History, d::Int64; debug::Int64 = 0)

    if d == 0
        return 0
    end

    a = alg.default_policy(pm, s)
    @assert isFeasible(pm, s, a)

    if debug > 2
        print(a, ", ")
    end

    s_, o, r = alg.Generative(pm, s, a)

    if isEnd(pm, s_)
        return r
    end

    return r + alg.rgamma_ * rollout_default(alg, pm, s_, h, d - 1, debug = debug)
end


function simulate(alg::POMCP, pm::POMDP, s::State, h::History, d::Int64; variant = nothing, MSState::Union(Dict{Any,Any}, Nothing) = nothing, bStat::Bool = false, debug::Int64 = 0)

    bSparseUCT = false
    sp_nObsMax = nothing

    bProgressiveWidening = false
    pw_c = nothing
    pw_alpha = nothing

    bUCB1_tuned = false

    bUCB_V = false
    uv_c = nothing

    bMSUCT = false
    ms_bPropagateN = false
    ms_L = nothing
    ms_N = nothing

    if variant != nothing
        if typeof(variant) <: Dict
            variant = {variant}
        end

        for variant_ in variant
            if variant_["type"] == :SparseUCT
                bSparseUCT = true
                sp_nObsMax = variant_["nObsMax"]
            elseif variant_["type"] == :ProgressiveWidening
                bProgressiveWidening = true
                pw_c = variant_["c"]
                pw_alpha = variant_["alpha"]
            elseif variant_["type"] == :UCB1_tuned
                bUCB1_tuned = true
            elseif variant_["type"] == :UCB_V
                bUCB_V = true
                uv_c = variant_["c"]
            elseif variant_["type"] == :MSUCT
                bMSUCT = true
                if haskey(variant_, "bPropagateN")
                    ms_bPropagateN = variant_["bPropagateN"]
                end
                ms_L = variant_["L"]
                ms_N = variant_["N"]
                if MSState == nothing
                    MSState = Dict{Any, Any}()
                    MSState["level"] = ones(Int64, pm.sc.nUAV)
                    MSState["nsplit"] = 0
                end
                MSState["nsplit"] = 1
            end
        end
    end

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

                alg.X2[(h, a)] = 0

                alg.Os[(h, a)] = Vector{Observation}[]
            end

            alg.Ns[h] = 1
            alg.T[h] = true
            alg.B[h] = [s]

        else
            push!(alg.B[h], s)

        end

        if debug > 2
            if d != 0
                println("    hit end")
            end
        end

        return 0
    end

    if !haskey(alg.T, h)
        if debug > 2
            println("    new node: ", h, " at level ", d)
        end

        for a in pm.actions
            alg.N[(h, a)] = 0

            if isFeasible(pm, s, a)
                alg.Q[(h, a)] = 0
            else
                alg.Q[(h, a)] = -Inf
            end

            alg.X2[(h, a)] = 0

            alg.Os[(h, a)] = Vector{Observation}[]
        end

        alg.Ns[h] = 1
        alg.T[h] = true
        alg.B[h] = [s]

        ro = alg.rollout_func(alg, pm, s, h, d, debug = debug)

        if debug > 2
            println("    rollout: ", neat(ro * pm.reward_norm_const))
        end

        return ro
    end

    if debug > 2
        println("    found node: ", h, " at level ", d)
    end

    Qv = Array(Float64, pm.nAction)
    Q = Array(Float64, pm.nAction)

    if bUCB1_tuned || bUCB_V
        Qv_ = Array(Float64, pm.nAction)
        var_ = zeros(pm.nAction)
        RE = zeros(pm.nAction)
    end

    for i = 1:pm.nAction
        a = pm.actions[i]

        Q[i] = alg.Q[(h, a)]

        if !isFeasible(pm, s, a)
            Qv[i] = -Inf
            if bUCB1_tuned || bUCB_V
                Qv_[i] = -Inf
            end
        elseif alg.N[(h, a)] == 0
            Qv[i] = Inf
            if bUCB1_tuned || bUCB_V
                Qv_[i] = Inf
            end
        else
            if bUCB1_tuned || bUCB_V
                if alg.N[(h, a)] > 1
                    var_[i] = (alg.X2[(h, a)] - alg.N[(h, a)] * (alg.Q[(h, a)] * alg.Q[(h, a)])) / (alg.N[(h, a)] - 1)
                    if abs(var_[i]) < 1.e-7
                        var_[i] = 0.
                    end
                    @assert var_[i] >= 0
                    RE[i] = sqrt(var_[i]) / abs(alg.Q[(h, a)])
                end

                Qv_[i] = alg.Q[(h, a)] + alg.c * sqrt(log(alg.Ns[h]) / alg.N[(h, a)])

                if bUCB1_tuned
                    Qv[i] = alg.Q[(h, a)] + sqrt(log(alg.Ns[h]) / alg.N[(h, a)] * min(1/4, var_[i] + sqrt(2 * log(alg.Ns[h]) / alg.N[(h, a)])))

                elseif bUCB_V
                    Qv[i] = alg.Q[(h, a)] + sqrt(2 * var_[i] * log(alg.Ns[h]) / alg.N[(h, a)]) + uv_c * 3 * log(alg.Ns[h]) / alg.N[(h, a)]

                end

            else
                Qv[i] = alg.Q[(h, a)] + alg.c * sqrt(log(alg.Ns[h]) / alg.N[(h, a)])

            end
        end
    end

    a = pm.actions[argmax(Qv)]

    # XXX need to backpropagate number of visits through intermediate nodes to root node?
    if bMSUCT
        MSState["nsplit"] = 0

        MSState_ = Dict{Any, Any}()
        MSState_["level"] = copy(MSState["level"])
        MSState_["nsplit"] = 0

        n = 1

        loc = [pm.cell_len / 2 + (s.location[1] - 1) * pm.cell_len, pm.cell_len / 2 + (s.location[2] - 1) * pm.cell_len]

        for i = 2:pm.sc.nUAV
            if MSState["level"][i] < length(ms_L) + 1
                loc_ = pm.sc_state.UAVStates[i].curr_loc

                if norm(loc - loc_) < ms_L[MSState["level"][i]]
                    if debug > 2
                        println("    UAV 1 ", loc, " and UAV ", i, " ", neat(loc_), " hit the level ", MSState["level"][i], " at level ", d)
                    end

                    if ms_N[MSState["level"][i]] > n
                        n = ms_N[MSState["level"][i]]
                    end

                    MSState_["level"][i] += 1
                end
            end
        end

        for i = 1:n
            s_, o, r = Generative(pm, s, a)

            if bSparseUCT
                if length(alg.Os[(h, a)]) < sp_nObsMax
                    push!(alg.Os[(h, a)], o)
                else
                    o = alg.Os[(h, a)][rand(1:length(alg.Os[(h, a)]))]
                end
            elseif bProgressiveWidening
                if length(alg.Os[(h, a)]) < ceil(pw_c * (alg.N[(h, a)] + 1) ^ pw_alpha)
                    push!(alg.Os[(h, a)], o)
                else
                    o = alg.Os[(h, a)][rand(1:length(alg.Os[(h, a)]))]
                end
            end

            if debug > 2
                println("    Q: ", neat(Q * pm.reward_norm_const), ", Qv: ", neat(Qv), ", (a, o): {", a, ", ", o, "}, s_: ", s_, ", r: ", neat(r * pm.reward_norm_const))
                if debug > 3
                    Na = zeros(Int64, pm.nAction)
                    for j = 1:pm.nAction
                        Na[j] = alg.N[(h, pm.actions[j])]
                    end
                    println("    Ns: ", alg.Ns[h], ", N: ", Na)
                end
                if i == n
                    if bUCB1_tuned || bUCB_V
                        println("    Qv_: ", neat(Qv_), ", RE: ", neat(RE))
                    end
                end
            end

            if alg.visualizer != nothing
                updateTree(alg.visualizer, :before_sim, s, a, o)
            end

            q = r + alg.gamma_ * simulate(alg, pm, s_, History([h.history, a, o]), d - 1, variant = variant, MSState = MSState_, debug = debug)

            if !ms_bPropagateN
                alg.N[(h, a)] += 1
                alg.Ns[h] += 1
            else
                alg.N[(h, a)] += MSState_["nsplit"]
                alg.Ns[h] += MSState_["nsplit"]
                MSState["nsplit"] += MSState_["nsplit"]
            end
            alg.Q[(h, a)] += (q - alg.Q[(h, a)]) / alg.N[(h, a)]
            alg.X2[(h, a)] += q * q

            if alg.visualizer != nothing
                updateTree(alg.visualizer, :after_sim, s, a, r * pm.reward_norm_const, q * pm.reward_norm_const, alg.N[(h, a)], alg.Ns[h], alg.Q[(h, a)] * pm.reward_norm_const)
            end
        end

    else
        s_, o, r = Generative(pm, s, a)

        if bSparseUCT
            if length(alg.Os[(h, a)]) < sp_nObsMax
                push!(alg.Os[(h, a)], o)
            else
                o = alg.Os[(h, a)][rand(1:length(alg.Os[(h, a)]))]
            end
        elseif bProgressiveWidening
            if length(alg.Os[(h, a)]) < ceil(pw_c * (alg.N[(h, a)] + 1) ^ pw_alpha)
                push!(alg.Os[(h, a)], o)
            else
                o = alg.Os[(h, a)][rand(1:length(alg.Os[(h, a)]))]
            end
        end

        if debug > 2
            println("    Q: ", neat(Q * pm.reward_norm_const), ", Qv: ", neat(Qv), ", (a, o): {", a, ", ", o, "}, s_: ", s_, ", r: ", neat(r * pm.reward_norm_const))
            if debug > 3
                Na = zeros(Int64, pm.nAction)
                for i = 1:pm.nAction
                    Na[i] = alg.N[(h, pm.actions[i])]
                end
                println("    Ns: ", alg.Ns[h], ", N: ", Na)
            end
            if bUCB1_tuned || bUCB_V
                println("    Qv_: ", neat(Qv_), ", RE: ", neat(RE))
            end
        end

        if alg.visualizer != nothing
            updateTree(alg.visualizer, :before_sim, s, a, o)
        end

        q = r + alg.gamma_ * simulate(alg, pm, s_, History([h.history, a, o]), d - 1, variant = variant, debug = debug)

        alg.N[(h, a)] += 1
        alg.Ns[h] += 1
        alg.Q[(h, a)] += (q - alg.Q[(h, a)]) / alg.N[(h, a)]
        alg.X2[(h, a)] += q * q

        if alg.visualizer != nothing
            updateTree(alg.visualizer, :after_sim, s, a, r * pm.reward_norm_const, q * pm.reward_norm_const, alg.N[(h, a)], alg.Ns[h], alg.Q[(h, a)] * pm.reward_norm_const)
        end

    end

    push!(alg.B[h], s)

    if bStat && d == alg.depth
        return q, a
    else
        return q
    end
end


function selectAction(alg::POMCP, pm::POMDP, b::Belief; variant = nothing, bStat::Bool = false, debug::Int64 = 0)

    if alg.visualizer != nothing
        initTree(alg.visualizer)
    end

    h = History()

    if !alg.reuse
        initialize(alg)
    end

    Q = Dict{Action, Float64}()
    for a in pm.actions
        Q[a] = 0.
    end

    if bStat
        Qs = Dict{Action, Vector{Float64}}()

        for a in pm.actions
            Qs[a] = Float64[]
        end
    end

    if alg.rollout_type == :CE_worst || alg.rollout_type == :CE_best
        alg.CE_samples = (Action, Int64)[]
    end

    start_time = time()

    for i = 1:alg.nloop_max
        if debug > 2
            println("  iteration: ", i)
        end

        s = sampleBelief(pm, b)

        if debug > 2
            println("  sample: ", s)
        end

        if !bStat
            simulate(alg, pm, s, h, alg.depth, variant = variant, debug = debug)
        else
            ret = simulate(alg, pm, s, h, alg.depth, variant = variant, bStat = true, debug = debug)
            if length(ret) == 2
                q, a = ret
                push!(Qs[a], q * pm.reward_norm_const)
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
            Q_prev = Q[a]
            Q[a] = alg.Q[(h, a)] * pm.reward_norm_const
            res += (Q[a] - Q_prev)^2
        end

        if i > alg.nloop_min &&  sqrt(res) < alg.eps
            break
        end

        if alg.runtime_max != 0 && time() - start_time > alg.runtime_max
            break
        end
    end

    Q_max = -Inf
    for a in pm.actions
        Q[a] = alg.Q[(h, a)] * pm.reward_norm_const

        if Q[a] > Q_max
            Q_max = Q[a]
        end
    end

    actions = Action[]
    for a in pm.actions
        if Q[a] == Q_max
            push!(actions, a)
        end
    end
    action = actions[rand(1:length(actions))]

    if alg.visualizer != nothing
        saveTree(alg.visualizer, pm)
    end

    if !bStat
        return action, Q
    else
        return action, Q, Qs
    end
end


function reinitialize(alg::POMCP, a::Action, o::Observation)

    T_new = Dict{History, Bool}()
    N_new = Dict{(History, Action), Int64}()
    Ns_new = Dict{History, Int64}()
    Q_new = Dict{(History, Action), Float64}()
    X2_new = Dict{(History, Action), Float64}()
    B_new = Dict{History, Vector{State}}()
    Os_new = Dict{(History, Action), Vector{Observation}}()

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
            X2_new[(History(h.history[3:end]), action)] = alg.X2[key]
            Os_new[(History(h.history[3:end]), action)] = alg.Os[key]
        end
    end

    alg.T = T_new
    alg.N = N_new
    alg.Ns = Ns_new
    alg.Q = Q_new
    alg.X2 = X2_new
    alg.B = B_new
    alg.Os = Os_new

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
    alg.X2 = Dict{(History, Action), Float64}()
    alg.B = Dict{History, Vector{State}}()
    alg.Os = Dict{(History, Action), Vector{Observation}}()

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


