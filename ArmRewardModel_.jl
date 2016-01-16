# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 12/14/2015

module ArmRewardModel_

export ArmRewardModel, sampleFromArmRewardModel, updateArmRewardModel


using Distributions
using ConjugatePriors

import ConjugatePriors: NormalGamma


type ArmRewardModel

    N::Int64

    V_alpha0::Union{Int64, Float64}
    V_beta0::Union{Int64, Float64}
    V::Int64

    mu0::Float64        # mu_0
    lambda0::Float64    # n_mu
    alpha0::Float64     # n_tau / 2
    beta0::Float64      # n_tau / (2 * tau0)

    mu::Float64
    lambda::Float64
    alpha::Float64
    beta_::Float64

    n::Int64
    q::Float64
    x2::Float64

    v_bound::Float64

    mu_v0::Float64      # mu_0
    lambda_v0::Float64  # n_mu
    alpha_v0::Float64   # n_tau / 2
    beta_v0::Float64    # n_tau / (2 * tau0)

    mu_v::Float64
    lambda_v::Float64
    alpha_v::Float64
    beta_v::Float64

    n_v::Int64
    q_v::Float64
    x2_v::Float64


    function ArmRewardModel(V_alpha0::Union{Int64, Float64}, V_beta0::Union{Int64, Float64}, mu0::Float64, lambda0::Float64, alpha0::Float64, beta0::Float64, v_bound::Float64, mu_v0::Float64, lambda_v0::Float64, alpha_v0::Float64, beta_v0::Float64)

        self = new()

        self.N = 0

        self.V_alpha0 = V_alpha0
        self.V_beta0 = V_beta0
        self.V = 0

        self.mu0 = mu0
        self.lambda0 = lambda0
        self.alpha0 = alpha0
        self.beta0 = beta0

        self.mu = mu0
        self.lambda = lambda0
        self.alpha = alpha0
        self.beta_ = beta0

        self.n = 0
        self.q = 0
        self.x2 = 0

        self.v_bound = v_bound

        self.mu_v0 = mu_v0
        self.lambda_v0 = lambda_v0
        self.alpha_v0 = alpha_v0
        self.beta_v0 = beta_v0

        self.mu_v = mu_v0
        self.lambda_v = lambda_v0
        self.alpha_v = alpha_v0
        self.beta_v = beta_v0

        self.n_v = 0
        self.q_v = 0
        self.x2_v = 0

        return self
    end
end


function sampleFromArmRewardModel(arm::ArmRewardModel)

    p = rand(Beta(arm.V + arm.V_alpha0, arm.N - arm.V + arm.V_beta0))

    mu, tau = rand(NormalGamma(arm.mu, arm.lambda, arm.alpha, arm.beta_))
    #r = rand(Normal(mu, sqrt(1 / tau)))

    mu_v, tau_v = rand(NormalGamma(arm.mu_v, arm.lambda_v, arm.alpha_v, arm.beta_v))
    #r_v = rand(Normal(mu_v, sqrt(1 / tau_v)))

    # expected reward
    #return (1 - p) * r + p * r_v

    # expected mean
    return (1 - p) * mu + p * mu_v
end


function updateArmRewardModel(arm::ArmRewardModel, q::Float64)

    arm.N += 1

    bV = false

    if q < arm.v_bound
        arm.V += 1

        bV = true
    end

    if !bV
        arm.n += 1
        arm.q += (q - arm.q) / arm.n
        arm.x2 += q * q

        if arm.n > 1
            s = 1 / arm.n * arm.x2 - arm.q^2

            if abs(s) < 1e-7
                s = 0.
            end

        else
            s = 0.

        end

        arm.mu = (arm.lambda0 * arm.mu0 + arm.n * arm.q) / (arm.lambda0 + arm.n)
        arm.lambda = arm.lambda0 + arm.n
        arm.alpha = arm.alpha0 + arm.n / 2
        arm.beta_ = arm.beta0 + 1 / 2 * (arm.n * s + arm.lambda0 * arm.n * (arm.q - arm.mu0)^2 / (arm.lambda0 + arm.n))

    else
        arm.n_v += 1
        arm.q_v += (q - arm.q_v) / arm.n_v
        arm.x2_v += q * q

        if arm.n_v > 1
            s = 1 / arm.n_v * arm.x2_v - arm.q_v^2

            if abs(s) < 1e-7
                s = 0.
            end

        else
            s = 0.

        end

        arm.mu_v = (arm.lambda_v0 * arm.mu_v0 + arm.n_v * arm.q_v) / (arm.lambda_v0 + arm.n_v)
        arm.lambda_v = arm.lambda_v0 + arm.n_v
        arm.alpha_v = arm.alpha_v0 + arm.n_v / 2
        arm.beta_v = arm.beta_v0 + 1 / 2 * (arm.n_v * s + arm.lambda_v0 * arm.n_v * (arm.q_v - arm.mu_v0)^2 / (arm.lambda_v0 + arm.n_v))

    end
end


end


