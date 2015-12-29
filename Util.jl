# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 08/04/2015

module Util

export argmax, sampleFromProb, neat, randi


using Base.Test


function argmax(A)

    max_ = maximum(A)
    indexes = find(A .== max_)
    index = indexes[rand(1:length(indexes))]

    return index
end


function sampleFromProb(p)

    p_cs = cumsum(p)
    @test_approx_eq p_cs[length(p)] 1.

    index = nothing
    rv = rand()

    for i = 1:length(p_cs)
        if rv < p_cs[i]
            index = i

            break
        end
    end

    return index
end


neat_(v) = round(signif(v, 4), 4)

function neat(v)

    if typeof(v) <: Array
        return map(x -> neat_(x), v)
    else
        return neat_(v)
    end
end


function randi(rng::AbstractRNG, range::UnitRange{Int64})

    rn = rand(rng)

    range_ = collect(range)

    boundary = collect(0:1/length(range_):1)[2:end]

    v = nothing

    for i = 1:length(boundary)
        if rn <= boundary[i]
            v = range_[i]
            break
        end
    end

    return v
end


end


