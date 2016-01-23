# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 11/04/2014

VERSION >= v"0.4" && __precompile__(false)


# Partially Observable Monte-Carlo Planning
# D. Silver and J. Veness, "Monte-Carlo Planning in Large POMDPs," in Advances in Neural Information Processing Systems (NIPS), 2010.

module POMCP_

export POMCP, selectAction, initialize, reinitialize, getParticles


using MCTS_
using TreePolicyLib
using POMDP_
using Util
using MCTSVisualizer_

using Base.Test


import MCTS_: selectAction, initialize, reinitialize


type TreePolicyParams

    bUCB1::Bool
    bUCB1_::Bool
    c::Float64

    bUCB1_tuned::Bool
    ut_c::Float64

    bUCB_V::Bool
    uv_c::Float64

    bUCBLike::Bool

    bTS::Bool

    bTSM::Bool
    arm_reward_model::Function

    bAUCB::Bool
    subpolicies::Vector{Dict{ASCIIString, Any}}
    control_policy::Dict{ASCIIString, Any}

    bUCBScale::Bool


    bSparseUCT::Bool
    sp_nObsMax::Int64

    bProgressiveWidening::Bool
    pw_c::Float64
    pw_alpha::Float64


    bMS::Bool
    ms_L::Vector{Float64}
    ms_N::Vector{Int64}


    function TreePolicyParams(tree_policies::Any = nothing)

        self = new()

        self.bUCB1 = false
        self.bUCB1_tuned = false
        self.bUCB_V = false

        self.bUCBLike = false

        self.bUCB1_ = false
        self.bTS = false
        self.bTSM = false
        self.bAUCB = false

        self.bUCBScale = false

        self.bSparseUCT = false
        self.bProgressiveWidening = false

        self.bMS = false

        if tree_policies == nothing
            self.bUCB1 = true
            self.bUCBLike = true
            self.c = sqrt(2)

        else
            if !(typeof(tree_policies) <: Array)
                tree_policies = Any[tree_policies]
            end

            for tree_policy in tree_policies
                if tree_policy["type"] == :UCB1
                    self.bUCB1 = true
                    self.bUCBLike = true
                    if haskey(tree_policy, "c")
                        self.c = tree_policy["c"]
                    else
                        self.c = sqrt(2)
                    end

                elseif tree_policy["type"] == :UCB1_
                    self.bUCB1_ = true
                    if haskey(tree_policy, "c")
                        self.c = tree_policy["c"]
                    else
                        self.c = sqrt(2)
                    end

                elseif tree_policy["type"] == :UCB1_tuned
                    self.bUCB1_tuned = true
                    self.bUCBLike = true
                    if haskey(tree_policy, "c")
                        self.ut_c = tree_policy["c"]
                    else
                        self.ut_c = 1/4
                    end

                elseif tree_policy["type"] == :UCB_V
                    self.bUCB_V = true
                    self.bUCBLike = true
                    self.uv_c = tree_policy["c"]

                elseif tree_policy["type"] == :TS
                    self.bTS = true

                elseif tree_policy["type"] == :TSM
                    self.bTSM = true
                    self.arm_reward_model = tree_policy["ARM"]

                elseif tree_policy["type"] == :AUCB
                    self.bAUCB = true
                    self.subpolicies = tree_policy["SP"]
                    if haskey(tree_policy, "CP")
                        self.control_policy = tree_policy["CP"]
                    else
                        self.control_policy = Dict("type" => :TSN)
                    end

                elseif tree_policy["type"] == :SparseUCT
                    self.bSparseUCT = true
                    self.sp_nObsMax = tree_policy["nObsMax"]

                elseif tree_policy["type"] == :ProgressiveWidening
                    self.bProgressiveWidening = true
                    self.pw_c = tree_policy["c"]
                    self.pw_alpha = tree_policy["alpha"]

                elseif tree_policy["type"] == :MS
                    self.bMS = true
                    self.ms_L = tree_policy["L"]
                    self.ms_N = tree_policy["N"]

                else
                    error("Unknown tree policy type, ", tree_policy["type"])

                end

                if haskey(tree_policy, "bScale")
                    self.bUCBScale = tree_policy["bScale"]
                end
            end
        end

        return self
    end
end


type POMCP <: MCTS

    seed::Union{Int64, Void}
    rng::AbstractRNG

    depth::Int64

    Generative::Function

    T::Dict{History, Bool}
    Ns::Dict{History, Int64}
    N::Dict{Tuple{History, Action}, Int64}
    Q::Dict{Tuple{History, Action}, Float64}

    X2::Dict{Tuple{History, Action}, Float64}

    B::Dict{History, Vector{State}}
    Os::Dict{Tuple{History, Action}, Vector{Observation}}

    TP::Dict{History, TreePolicy}

    nloop_max::Int64
    nloop_min::Int64
    eps::Float64

    runtime_max::Float64

    gamma_::Float64

    tree_policy::TreePolicyParams

    rollout_type::Symbol
    rollout_func::Function

    CE_samples::Vector{Tuple{Action, Float64}}
    CE_dist_param::Vector{Float64}

    bReuse::Bool

    visualizer::Union{MCTSVisualizer, Void}


    function POMCP(;seed::Union{Int64, Void} = nothing, depth::Int64 = 3, nloop_max::Int64 = 1000, nloop_min::Int64 = 100, eps::Float64 = 1.e-3, runtime_max::Float64 = 0., gamma_::Float64 = 0.9, tree_policy::Any = nothing, rollout::Union{Tuple{Symbol, Function}, Void} = nothing, bReuse::Bool = true, visualizer::Union{MCTSVisualizer, Void} = nothing)

        self = new()

        if seed == nothing
            self.seed = round(Int64, time())
        else
            self.seed = seed
        end

        self.rng = MersenneTwister(self.seed)

        self.depth = depth

        self.Generative = Generative

        self.T = Dict{History, Bool}()
        self.Ns = Dict{History, Int64}()
        self.N = Dict{Tuple{History, Action}, Int64}()
        self.Q = Dict{Tuple{History, Action}, Float64}()

        self.X2 = Dict{Tuple{History, Action}, Float64}()

        self.B = Dict{History, Vector{State}}()
        self.Os = Dict{Tuple{History, Action}, Vector{Observation}}()

        self.TP = Dict{History, TreePolicy}()

        self.nloop_max = nloop_max
        self.nloop_min = nloop_min
        self.eps = eps

        self.runtime_max = runtime_max

        self.gamma_ = gamma_

        self.tree_policy = TreePolicyParams(tree_policy)

        if rollout == nothing
            self.rollout_type = :default
            self.rollout_func = rollout_default
        else
            self.rollout_type = rollout[1]
            self.rollout_func = rollout[2]
        end

        self.bReuse = bReuse

        self.visualizer = visualizer

        return self
    end
end


function Generative(pm::POMDP, s::State, a::Action)

    s_, o, r = POMDP_.Generative(pm, s, a)

    return s_, o, r / pm.reward_norm_const
end


function default_policy(pm::POMDP, s::State)
    
    # Note: pass pm.rng to rand() if pm supports rng

    a = pm.actions[rand(1:pm.nActions)]

    while !isFeasible(pm, s, a)
        a = pm.actions[rand(1:pm.nActions)]
    end

    return a
end


function rollout_default(alg::POMCP, pm::POMDP, s::State, h::History, d::Int64; rgamma::Float64 = 0.9, debug::Int64 = 0)

    if d == 0 || isEnd(pm, s)
        return 0
    end

    a = default_policy(pm, s)

    if debug > 3
        print(string(a), ", ")
    end

    s_, o, r = alg.Generative(pm, s, a)

    return r + rgamma * rollout_default(alg, pm, s_, h, d - 1, rgamma = rgamma, debug = debug)
end


function simulate(alg::POMCP, pm::POMDP, s::State, h::History, d::Int64; MSState::Union{Dict{ASCIIString,Any}, Void} = nothing, bStat::Bool = false, debug::Int64 = 0)

    if alg.tree_policy.bMS
        if MSState == nothing
            MSState = Dict{ASCIIString, Any}()
            MSState["level"] = ones(Int64, pm.sc.nUAV)
        end
    end

    if alg.visualizer != nothing
        updateTree(alg.visualizer, :start_sim, s)
    end

    bEnd = isEnd(pm, s)

    if d == 0 || bEnd
        if !haskey(alg.T, h)
            for a in pm.actions
                alg.N[(h, a)] = 0
                alg.Q[(h, a)] = 0
                alg.X2[(h, a)] = 0
                alg.Os[(h, a)] = Vector{Observation}[]
            end

            if !alg.tree_policy.bUCBLike
                if alg.tree_policy.bUCB1_
                    alg.TP[h] = UCB1Policy(pm, c = alg.tree_policy.c, bScale = alg.tree_policy.bUCBScale)
                elseif alg.tree_policy.bTS
                    alg.TP[h] = TSPolicy(pm)
                elseif alg.tree_policy.bTSM
                    alg.TP[h] = TSMPolicy(pm, alg.tree_policy.arm_reward_model)
                elseif alg.tree_policy.bAUCB
                    alg.TP[h] = AUCBPolicy(pm, alg.tree_policy.subpolicies, control_policy = alg.tree_policy.control_policy, bScale = alg.tree_policy.bUCBScale)
                end
            end

            alg.T[h] = true
            alg.Ns[h] = 0
            alg.B[h] = [s]

        else
            push!(alg.B[h], s)

        end

        if bEnd && debug > 2
            println("    hit end")
        end

        if d == 0
            ro = alg.rollout_func(alg, pm, s, h, d, debug = debug)

            if debug > 2
                println("    rollout: ", neat(ro * pm.reward_norm_const))
            end

            return ro
        end

        return 0
    end

    if !haskey(alg.T, h)
        if debug > 2
            println("    new node: ", string(h), " at level ", d)
        end

        for a in pm.actions
            alg.N[(h, a)] = 0
            alg.Q[(h, a)] = 0
            alg.X2[(h, a)] = 0
            alg.Os[(h, a)] = Vector{Observation}[]
        end

        if !alg.tree_policy.bUCBLike
            if alg.tree_policy.bUCB1_
                alg.TP[h] = UCB1Policy(pm, c = alg.tree_policy.c, bScale = alg.tree_policy.bUCBScale)
            elseif alg.tree_policy.bTS
                alg.TP[h] = TSPolicy(pm)
            elseif alg.tree_policy.bTSM
                alg.TP[h] = TSMPolicy(pm, alg.tree_policy.arm_reward_model)
            elseif alg.tree_policy.bAUCB
                alg.TP[h] = AUCBPolicy(pm, alg.tree_policy.subpolicies, control_policy = alg.tree_policy.control_policy, bScale = alg.tree_policy.bUCBScale)
            end
        end

        alg.T[h] = true
        alg.Ns[h] = 0
        alg.B[h] = [s]

        if debug > 3 && alg.rollout_type == :default
            print("    ")
        end

        if alg.rollout_type == :MS
            ro = alg.rollout_func(alg, pm, s, h, d, MSState = deepcopy(MSState), debug = debug)
        else
            ro = alg.rollout_func(alg, pm, s, h, d, debug = debug)
        end

        if debug > 3 && alg.rollout_type == :default
            println()
        end

        if debug > 2
            println("    rollout: ", neat(ro * pm.reward_norm_const))
        end

        return ro
    end

    if debug > 2
        println("    found node: ", string(h), " at level ", d)
    end

    Q = Array(Float64, pm.nActions)
    for i = 1:pm.nActions
        a = pm.actions[i]
        Q[i] = alg.Q[(h, a)]
    end

    Qv = Array(Float64, pm.nActions)

    if alg.tree_policy.bUCBLike
        if alg.tree_policy.bUCB1_tuned || alg.tree_policy.bUCB_V
            var_ = zeros(pm.nActions)
            RE = zeros(pm.nActions)
        end

        for i = 1:pm.nActions
            a = pm.actions[i]

            if !isFeasible(pm, s, a)
                Qv[i] = -Inf

            elseif alg.N[(h, a)] == 0
                Qv[i] = Inf

            else
                if alg.tree_policy.bUCB1
                    if !alg.tree_policy.bUCBScale
                        Qv[i] = alg.Q[(h, a)] + alg.tree_policy.c * sqrt(log(alg.Ns[h]) / alg.N[(h, a)])
                    else
                        Qv[i] = alg.Q[(h, a)] + alg.tree_policy.c * sqrt((pm.reward_max - pm.reward_min) * d) * sqrt(log(alg.Ns[h]) / alg.N[(h, a)])
                    end

                elseif alg.tree_policy.bUCB1_tuned || alg.tree_policy.bUCB_V
                    if alg.N[(h, a)] > 1
                        var_[i] = (alg.X2[(h, a)] - alg.N[(h, a)] * (alg.Q[(h, a)] * alg.Q[(h, a)])) / (alg.N[(h, a)] - 1)
                        if abs(var_[i]) < 1.e-7
                            var_[i] = 0.
                        end
                        @assert var_[i] >= 0.
                        RE[i] = sqrt(var_[i]) / abs(alg.Q[(h, a)])
                    end

                    if alg.tree_policy.bUCB1_tuned
                        Qv[i] = alg.Q[(h, a)] + sqrt(log(alg.Ns[h]) / alg.N[(h, a)] * min(alg.tree_policy.ut_c, var_[i] + sqrt(2 * log(alg.Ns[h]) / alg.N[(h, a)])))

                    elseif alg.tree_policy.bUCB_V
                        Qv[i] = alg.Q[(h, a)] + sqrt(2 * var_[i] * log(alg.Ns[h]) / alg.N[(h, a)]) + alg.tree_policy.uv_c * 3 * log(alg.Ns[h]) / alg.N[(h, a)]

                    end

                end

            end
        end

        a = pm.actions[argmax(Qv)]

    else
        feasible_actions = Dict{Action, Bool}()

        for a in pm.actions
            if isFeasible(pm, s, a)
                feasible_actions[a] = true
            else
                feasible_actions[a] = false
            end
        end

        if alg.tree_policy.bAUCB
            a, Qv, sindex = TreePolicyLib.selectAction(alg.TP[h], pm, feasible_actions = feasible_actions, d = d)
        else
            a, Qv = TreePolicyLib.selectAction(alg.TP[h], pm, feasible_actions = feasible_actions, d = d)
        end

    end

    n = 1

    if alg.tree_policy.bMS
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
    end

    q_ = 0

    for k = 1:n
        s_, o, r = Generative(pm, s, a)

        if alg.tree_policy.bSparseUCT
            if length(alg.Os[(h, a)]) < alg.tree_policy.sp_nObsMax
                push!(alg.Os[(h, a)], o)
            else
                o = alg.Os[(h, a)][randi(alg.rng, 1:length(alg.Os[(h, a)]))]
            end
        elseif alg.tree_policy.bProgressiveWidening
            if length(alg.Os[(h, a)]) < ceil(alg.tree_policy.pw_c * (alg.N[(h, a)] + 1) ^ alg.tree_policy.pw_alpha)
                push!(alg.Os[(h, a)], o)
            else
                o = alg.Os[(h, a)][randi(alg.rng, 1:length(alg.Os[(h, a)]))]
            end
        end

        if debug > 2
            println("    Q: ", neat(Q * pm.reward_norm_const), ", Qv: ", neat(Qv), ", a: ", string(a), ", s_: ", string(s_), ", o: ", string(o), ", r: ", neat(r * pm.reward_norm_const))
            if k == n
                if debug > 3
                    Na = zeros(Int64, pm.nActions)
                    for i = 1:pm.nActions
                        Na[i] = alg.N[(h, pm.actions[i])]
                    end
                    println("    Ns: ", alg.Ns[h], ", N: ", Na)
                end
                if alg.tree_policy.bUCB1_tuned || alg.tree_policy.bUCB_V
                    println("    var: ", neat(var_), ", RE: ", neat(RE))
                end
            end
        end

        if alg.visualizer != nothing
            updateTree(alg.visualizer, :before_sim, s, a, o)
        end

        if alg.tree_policy.bMS
            q = r + alg.gamma_ * simulate(alg, pm, s_, History([h.history; a; o]), d - 1, MSState = deepcopy(MSState), debug = debug)
            q_ += (q - q_) / k
        else
            q = r + alg.gamma_ * simulate(alg, pm, s_, History([h.history; a; o]), d - 1, debug = debug)
            q_ = q
        end

        alg.Ns[h] += 1
        alg.N[(h, a)] += 1
        alg.Q[(h, a)] += (q - alg.Q[(h, a)]) / alg.N[(h, a)]
        alg.X2[(h, a)] += q * q

        if !alg.tree_policy.bUCBLike
            if alg.tree_policy.bAUCB
                updatePolicy(alg.TP[h], pm, a, q, sindex)
            else
                updatePolicy(alg.TP[h], pm, a, q)
            end
        end

        if alg.visualizer != nothing
            updateTree(alg.visualizer, :after_sim, s, a, r * pm.reward_norm_const, q * pm.reward_norm_const, alg.Ns[h], alg.N[(h, a)], alg.Q[(h, a)] * pm.reward_norm_const)
        end
    end

    push!(alg.B[h], s)

    if bStat && d == alg.depth
        return q_, a
    else
        return q_
    end
end


function selectAction(alg::POMCP, pm::POMDP, b::Belief; bStat::Bool = false, debug::Int64 = 0)

    if alg.visualizer != nothing
        initTree(alg.visualizer)
    end

    h = History()

    if !alg.bReuse
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

    start_time = time()

    n = 0
    res = 0.

    for i = 1:alg.nloop_max
        if debug > 2
            println("  iteration: ", i)
        end

        n = i

        s = sampleBelief(pm, b)

        if debug > 2
            println("  sample: ", string(s))
        end

        if !bStat
            simulate(alg, pm, s, h, alg.depth, debug = debug)
        else
            ret = simulate(alg, pm, s, h, alg.depth, bStat = true, debug = debug)
            if length(ret) == 2
                q, a = ret
                push!(Qs[a], q * pm.reward_norm_const)
            end
        end

        #println("h: ", h)
        #println("T: ", alg.T)
        #println("Ns: ", alg.Ns)
        #println("N: ", alg.N)
        #println("Q: ", alg.Q)
        #println()

        res = 0.

        for a in pm.actions
            Q_prev = Q[a]
            Q[a] = alg.Q[(h, a)] * pm.reward_norm_const
            if !isinf(Q[a])
                res += (Q[a] - Q_prev)^2
            end
        end

        if i >= alg.nloop_min
            if sqrt(res) < alg.eps
                break
            elseif alg.runtime_max != 0 && time() - start_time > alg.runtime_max
                break
            end
        end
    end

    if debug > 1
        println("  # of iterations: ", n)
        println("  residual: ", sqrt(res))
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
    action = actions[randi(alg.rng, 1:length(actions))]

    if alg.visualizer != nothing
        saveTree(alg.visualizer, pm)
    end

    if !bStat
        return action, Q
    else
        return action, Q, Qs
    end
end


function initialize(alg::POMCP)

    alg.T = Dict{History, Bool}()
    alg.Ns = Dict{History, Int64}()
    alg.N = Dict{Tuple{History, Action}, Int64}()
    alg.Q = Dict{Tuple{History, Action}, Float64}()

    alg.X2 = Dict{Tuple{History, Action}, Float64}()

    alg.B = Dict{History, Vector{State}}()
    alg.Os = Dict{Tuple{History, Action}, Vector{Observation}}()

    alg.TP= Dict{History, TreePolicy}()

    alg.bReuse = false

    if alg.visualizer != nothing
        alg.visualizer.b_hist_acc = false
    end
end


function reinitialize(alg::POMCP, a::Action, o::Observation)

    T_new = Dict{History, Bool}()
    Ns_new = Dict{History, Int64}()
    N_new = Dict{Tuple{History, Action}, Int64}()
    Q_new = Dict{Tuple{History, Action}, Float64}()

    X2_new = Dict{Tuple{History, Action}, Float64}()

    B_new = Dict{History, Vector{State}}()
    Os_new = Dict{Tuple{History, Action}, Vector{Observation}}()

    TP_new = Dict{History, TreePolicy}()

    for h in keys(alg.T)
        if length(h.history) > 0 && h.history[1] == a && h.history[2] == o
            T_new[History(h.history[3:end])] = alg.T[h]
            Ns_new[History(h.history[3:end])] = alg.Ns[h]
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

    if !alg.tree_policy.bUCBLike
        for key in keys(alg.N)
            h, _ = key
            if length(h.history) > 0 && h.history[1] == a && h.history[2] == o
                TP_new[History(h.history[3:end])] = alg.TP[h]
            end
        end
    end

    alg.T = T_new
    alg.Ns = Ns_new
    alg.N = N_new
    alg.Q = Q_new

    alg.X2 = X2_new

    alg.B = B_new
    alg.Os = Os_new

    alg.TP = TP_new

    alg.bReuse = true

    if alg.visualizer != nothing
        alg.visualizer.b_hist_acc = true
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


