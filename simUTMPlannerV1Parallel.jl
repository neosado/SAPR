# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 01/15/2016

function init_cluster(parallel::Symbol = :local_)

    ncpu_local = CPU_CORES / 2
    machines = [
        ("youngjun@tula", 20, "/home/youngjun/SAPR"),
        ("youngjun@cheonan", 4, "/home/youngjun/SAPR"),
        ("youngjun@cambridge", 6, "/home/youngjun/SAPR")
    ]

    if parallel == :local_ || parallel == :both
        addprocs(round(Int64, ncpu_local))
    end

    if parallel == :remote || parallel == :both
        for (machine, count, dir) in machines
            cluster_list = ASCIIString[]

            for i = 1:count
                push!(cluster_list, machine)
            end

            addprocs(cluster_list, dir = dir)
        end
    end
end


bParallel = true
parallel = :local_

bAppend = false

if "parallel" in ARGS
    bParallel = true
elseif "serial" in ARGS
    bParallel = false
end

if "local" in ARGS
    parallel = :local_
elseif "remote" in ARGS
    parallel = :remote
elseif "both" in ARGS
    parallel = :both
end

if "append" in ARGS
    bAppend = true
end

if bParallel
    init_cluster(parallel)
end

push!(LOAD_PATH, ".")
using simUTMPlannerV1Mod

runExpBatch(bParallel = bParallel, bAppend = bAppend)


