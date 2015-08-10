module Util

export argmax, sampleFromProb


using Base.Test


function argmax(A)

    max_ = maximum(A)
    indexes = find(A .== max_)
    index = indexes[rand(1:length(indexes))]

    return index
end


function sampleFromProb(p)

    p_cs = cumsum(p)
    @test p_cs[length(p)] == 1.

    index = -1
    rv = rand()

    for i = 1:length(p_cs)
        if rv < p_cs[i]
            index = i

            break
        end
    end

    return index
end


end


