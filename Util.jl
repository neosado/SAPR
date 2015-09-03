module Util

export argmax, sampleFromProb, neat


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


end


