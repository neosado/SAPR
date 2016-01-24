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


function init_cluster_sherlock()

    if !haskey(ENV, "SLURM_JOB_NODELIST")
        error("SLURM_JOB_NODELIST not defined")
    end

    if !haskey(ENV, "SLURM_NTASKS_PER_NODE")
        error("SLURM_NTASKS_PER_NODE not defined")
    end

    n = parse(Int64, ENV["SLURM_NTASKS_PER_NODE"])
    dir = "/home/youngjun/SAPR"

    machines = Tuple{ASCIIString, Int64}[]

    list = ENV["SLURM_JOB_NODELIST"]

    for m = eachmatch(r"([\w\d-]+)(\[[\d,-]+\])?", list)
        if m.captures[2] == nothing
            push!(machines, (m.captures[1], n))

        else
            host_pre = m.captures[1]

            s = split(m.captures[2][2:end-1], ",")
            for s_ in s
                if isdigit(s_)
                    push!(machines, (host_pre * s_, n))
                else
                    a, b = split(s_, "-")
                    for i = parse(Int64, a):parse(Int64, b)
                        push!(machines, (host_pre * string(i), n))
                    end
                end
            end

        end
    end

    for (machine, count) in machines
        cluster_list = ASCIIString[]

        for i = 1:count
            push!(cluster_list, machine)
        end

        addprocs(cluster_list, dir = dir)
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
    if contains(gethostname(), "sherlock")
        init_cluster_sherlock()
    else
        init_cluster(parallel)
    end
end

push!(LOAD_PATH, ".")
using simUTMPlannerV1Mod

runExpBatch(bParallel = bParallel, bAppend = bAppend)


