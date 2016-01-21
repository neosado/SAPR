# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 12/14/2015

VERSION >= v"0.4" && __precompile__(false)


module TreePolicyLib

export TreePolicy, UCB1Policy, TSPolicy, TSMPolicy, AUCBPolicy
export selectAction, updatePolicy


using POMDP_
using ArmRewardModel_

using Distributions
using Base.Test
using Util


abstract TreePolicy


#
# UCB1
#

type UCB1Policy <: TreePolicy

    c::Float64

    bScale::Bool

    N_total::Int64
    N::Dict{Action, Int64}
    Q::Dict{Action, Float64}
    

    function UCB1Policy(pm::POMDP; c::Float64 = sqrt(2), bScale::Bool = false)

        self = new()

        self.c = c

        self.bScale = bScale

        if bScale
            @test pm.reward_min != -Inf
            @test pm.reward_max != Inf
        end

        self.N_total = 0
        self.N = Dict{Action, Int64}()
        self.Q = Dict{Action, Float64}()

        for a in pm.actions
            self.N[a] = 0
            self.Q[a] = 0.
        end
        
        return self
    end
end

function selectAction(policy::UCB1Policy, pm::POMDP; feasible_actions::Union{Dict{Action, Bool}, Void} = nothing, d::Union{Int64, Void} = nothing)

    Qv = zeros(pm.nActions)

    for i = 1:pm.nActions
        if feasible_actions != nothing && !feasible_actions[pm.actions[i]]
            Qv[i] = -Inf
        elseif policy.N[pm.actions[i]] == 0
            Qv[i] = Inf
        else
            if !policy.bScale
                Qv[i] = policy.Q[pm.actions[i]] + policy.c * sqrt(log(policy.N_total) / policy.N[pm.actions[i]])
            else
                @test d != nothing
                Qv[i] = policy.Q[pm.actions[i]] + policy.c * sqrt((pm.reward_max - pm.reward_min) * d) * sqrt(log(policy.N_total) / policy.N[pm.actions[i]])
            end
        end
    end

    k = argmax(Qv)

    return pm.actions[k], Qv
end

function updatePolicy(policy::UCB1Policy, pm::POMDP, a::Action, q::Float64)

    policy.N_total += 1
    policy.N[a] += 1
    policy.Q[a] += (q - policy.Q[a]) / policy.N[a]
end


#
# Thompson Sampling
#

type TSPolicy <: TreePolicy

    S::Dict{Action, Int64}
    F::Dict{Action, Int64}
    

    function TSPolicy(pm::POMDP)
        
        @test pm.reward_min != -Inf
        @test pm.reward_max != Inf

        self = new()

        self.S = Dict{Action, Int64}()
        self.F = Dict{Action, Int64}()

        for a in pm.actions
            self.S[a] = 0
            self.F[a] = 0
        end
        
        return self
    end
end

function selectAction(policy::TSPolicy, pm::POMDP; feasible_actions::Union{Dict{Action, Bool}, Void} = nothing, d::Union{Int64, Void} = nothing)

    theta = zeros(pm.nActions)

    for i = 1:pm.nActions
        if feasible_actions != nothing && !feasible_actions[pm.actions[i]]
            theta[i] = -Inf
        else
            theta[i] = rand(Beta(policy.S[pm.actions[i]] + 1, policy.F[pm.actions[i]] + 1))
        end
    end

    k = argmax(theta)

    return pm.actions[k], theta
end

function updatePolicy(policy::TSPolicy, pm::POMDP, a::Action, q::Float64)

    if q < pm.reward_min
        q = pm.reward_min
    elseif q > pm.reward_max
        q = pm.reward_max
    end

    q_ = (q - pm.reward_min) / (pm.reward_max - pm.reward_min)

    if rand(Bernoulli(q_)) == 1
        policy.S[a] += 1
    else
        policy.F[a] += 1
    end
end


#
# Thompson Sampling with Gaussian Mixture Model
#

type TSMPolicy <: TreePolicy

    ARM::Dict{Action, ArmRewardModel}
    

    function TSMPolicy(pm::POMDP, arm_reward_model::Function)

        self = new()

        self.ARM = Dict{Action, ArmRewardModel}()

        for a in pm.actions
            self.ARM[a] = arm_reward_model()
        end
        
        return self
    end
end

function selectAction(policy::TSMPolicy, pm::POMDP; feasible_actions::Union{Dict{Action, Bool}, Void} = nothing, d::Union{Int64, Void} = nothing)

    theta = zeros(pm.nActions)

    for i = 1:pm.nActions
        if feasible_actions != nothing && !feasible_actions[pm.actions[i]]
            theta[i] = -Inf
        else
            theta[i] = sampleFromArmRewardModel(policy.ARM[pm.actions[i]])
        end
    end

    k = argmax(theta)

    return pm.actions[k], theta
end

function updatePolicy(policy::TSMPolicy, pm::POMDP, a::Action, q::Float64)

    updateArmRewardModel(policy.ARM[a], q)
end


#
# Adaptive UCB
#

type AUCBPolicy <: TreePolicy

    control_policy::Dict{ASCIIString, Any}

    subpolicies::Array{TreePolicy}
    nSubpolicies::Int64

    bScale::Bool

    N_total::Int64
    N::Vector{Int64}
    Q::Vector{Float64}
    X2::Vector{Float64}
    

    function AUCBPolicy(pm::POMDP, subpolicies::Vector{Dict{ASCIIString, Any}}; control_policy::Dict{ASCIIString, Any} = Dict("type" => :TSN), bScale::Bool = false)

        self = new()

        self.control_policy = control_policy

        self.nSubpolicies = length(subpolicies)
        self.subpolicies = Array(TreePolicy, self.nSubpolicies)

        self.bScale = bScale

        i = 1
        for subpolicy in subpolicies
            if subpolicy["type"] == :UCB1
                if haskey(subpolicy, "c")
                    c = Float64(subpolicy["c"])
                else
                    c = sqrt(2)
                end
                self.subpolicies[i] = UCB1Policy(pm, c = c, bScale = bScale)

            elseif subpolicy["type"] == :TS
                self.subpolicies[i] = TSPolicy(pm)

            elseif subpolicy["type"] == :TSM
                self.subpolicies[i] = TSMPolicy(pm, subpolicy["ARM"])

            else
                error("Unknown subpolicy type, ", subpolicy["type"])

            end

            i += 1
        end

        self.N_total = 0
        self.N = zeros(Int64, self.nSubpolicies)
        self.Q = zeros(self.nSubpolicies)
        self.X2 = zeros(self.nSubpolicies)
        
        return self
    end
end

function selectAction(policy::AUCBPolicy, pm::POMDP; feasible_actions::Union{Dict{Action, Bool}, Void} = nothing, d::Union{Int64, Void} = nothing)

    Qv = zeros(policy.nSubpolicies)

    if policy.control_policy["type"] == :UCB1
        if haskey(policy.control_policy, "c")
            c = policy.control_policy["c"]
        else
            c = sqrt(2)
        end

        for i = 1:nSubpolicies
            if policy.N[i] == 0
                Qv[i] = Inf
            else
                if !policy.bScale
                    Qv[i] = policy.Q[i] + c * sqrt(log(policy.N_total) / policy.N[i])
                else
                    @test d != nothing
                    Qv[i] = policy.Q[i] + c * sqrt((pm.reward_max - pm.reward_min) * d) * sqrt(log(policy.N_total) / policy.N[i])
                end
            end
        end

    elseif policy.control_policy["type"] == :TSN
        if haskey(policy.control_policy, "c")
            c = policy.control_policy["c"]
        else
            c = 1.
        end

        var_ = zeros(policy.nSubpolicies)
        sigma = zeros(policy.nSubpolicies)

        for i = 1:policy.nSubpolicies
            if policy.N[i] < 2
                Qv[i] = Inf
            else
                var_[i] = (policy.X2[i] - policy.N[i] * (policy.Q[i] * policy.Q[i])) / (policy.N[i] - 1)
                if abs(var_[i]) < 1.e-3
                    var_[i] = 0.
                end

                if var_[i] == 0.
                    Qv[i] = policy.Q[i]
                else
                    sigma[i] = c * sqrt(var_[i] / policy.N[i])
                    Qv[i] = rand(Normal(policy.Q[i], sigma[i]))
                end
            end
        end

    else
        error("Unknown control policy type, ", policy.control_policy["type"])

    end

    subpolicy_index = argmax(Qv)

    return selectAction(policy.subpolicies[subpolicy_index], pm, feasible_actions = feasible_actions, d = d)..., subpolicy_index
end

function updatePolicy(policy::AUCBPolicy, pm::POMDP, a::Action, q::Float64, subpolicy_index::Int64)

    policy.N_total += 1
    policy.N[subpolicy_index] += 1
    policy.Q[subpolicy_index] += (q - policy.Q[subpolicy_index]) / policy.N[subpolicy_index]
    policy.X2[subpolicy_index] += q * q

    updatePolicy(policy.subpolicies[subpolicy_index], pm, a, q)
end


end


