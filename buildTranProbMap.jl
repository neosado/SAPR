using Distributions

using Util


function coord2grid(coord::Vector{Float64}, cell_len::Float64)

    x = coord[1]
    y = coord[2]

    if x == 0.
        xg = 1
    else
        xg = int64(ceil(x / cell_len))
    end

    if y == 0.
        yg = 1
    else
        yg = int64(ceil(y / cell_len))
    end

    return [xg, yg]
end


function grid2coord(grid::Vector{Float64}, cell_len::Float64)

    xg, yg = grid

    x = cell_len / 2 + (xg - 1 ) * cell_len
    y = cell_len / 2 + (yg - 1 ) * cell_len

    return [x, y]
end


velocity = 40

dt = 1

loc_err_sigma = 200.

cell_len = 30.

t_max = 1

eps = 1e-4


ngridx = int64(ceil(velocity * t_max / cell_len))
ngridy = 2 * ngridx + 1

PMap = zeros(ngridx, ngridy)

n = 1
PMap_ = copy(PMap)

while true
    curr_loc = [cell_len / 2, cell_len / 2 + floor(ngridy / 2) * cell_len]
    hloc = [1000., cell_len / 2 + floor(ngridy / 2) * cell_len]

    for t = 1:t_max
        loc_estimate = rand(MvNormal(curr_loc, loc_err_sigma))
        curr_loc += (hloc - loc_estimate) / norm(hloc - loc_estimate) * velocity * dt
    end

    xg, yg = coord2grid(curr_loc, cell_len)

    if xg < 1 || xg > ngridx || yg < 1 || yg > ngridy
        println(curr_loc)
        println([xg, yg])
    end
    PMap[xg, yg] += 1

    if n % 10000 == 0
        res = norm((PMap - PMap_) / n)

        println(n, ", ", res)

        if res < eps
            break
        end

        PMap_ = copy(PMap)
    end

    n += 1
end


println(n)
println(PMap)
println(neat(PMap / n))


