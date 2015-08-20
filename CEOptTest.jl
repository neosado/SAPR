using Distributions

y = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

n = 10
p = ones(n) / 2

N = 50
rho = 0.1

quiet = false

println("y: ", y')
println()
println("N: ", N)
println("rho: ", rho)
println()

gamma = 0
count = 0

distX = Array(DiscreteUnivariateDistribution, n)
X = Array(Float64, N, n)
S = Array(Float64, N)

t = 1

while true
    for i = 1:n
        distX[i] = Bernoulli(p[i])
    end

    for i = 1:N
        for j = 1:n
            X[i, j] = rand(distX[j])
        end

        S[i] = n - sum(abs(X[i, :]' - y))
    end

    Ssorted = sort(S)

    gamma_t = Ssorted[ceil((1 - rho) * N)]

    I = map((x) -> x >= gamma_t ? 1 : 0, S)

    for j = 1:n
        I_ = map((x) -> x == 1? 1 : 0, X[:, j])
        p[j] = sum(I .* I_) / sum(I)
    end

    if gamma_t == gamma
        count += 1
    else
        count = 0
    end

    if count == 1
        break
    end

    gamma = gamma_t

    if !quiet
        println("Number of steps: $t")
        println("gamma: $gamma")
        println("p: ", p')
        println()
    end

    t += 1
end

println("Number of steps: $t")
println("gamma: $gamma")
println("p: ", p')


