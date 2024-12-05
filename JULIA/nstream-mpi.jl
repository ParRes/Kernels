#
# Copyright (c) 2020, Intel Corporation
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above
#      copyright notice, this list of conditions and the following
#      disclaimer in the documentation and/or other materials provided
#      with the distribution.
# * Neither the name of Intel Corporation nor the names of its
#      contributors may be used to endorse or promote products
#      derived from this software without specific prior written
#      permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

#*******************************************************************
#
# NAME:    nstream
#
# PURPOSE: To compute memory bandwidth when adding a vector of a given
#          number of double precision values to the scalar multiple of
#          another vector of the same length, and storing the result in
#          a third vector.
#
# USAGE:   The program takes as input the number
#          of iterations to loop over the triad vectors, the length of the
#          vectors, and the offset between vectors
#
#          <progname> <# iterations> <vector length> <offset>
#
#          The output consists of diagnostics to make sure the
#          algorithm worked, and of timing statistics.
#
# NOTES:   Bandwidth is determined as the number of words read, plus the
#          number of words written, times the size of the words, divided
#          by the execution time. For a vector length of N, the total
#          number of words read and written is 4*N*sizeof(double).
#
#
# HISTORY: This code is loosely based on the Stream benchmark by John
#          McCalpin, but does not follow all the Stream rules. Hence,
#          reported results should not be associated with Stream in
#          external publications
#
#          Converted to Python by Jeff Hammond, October 2017.
#          Converted to Julia by ???
#          Improved by Carsten Bauer, November 2024.
#
# *******************************************************************

import MPI

# ********************************************************************
# read and test input parameters
# ********************************************************************

function do_initialize!(A, B, C, N)
    for i in eachindex(A, B, C)
        A[i] = 0.0
        B[i] = 2.0
        C[i] = 2.0
    end
end

function do_nstream!(A, B, C, s, N)
    for i in eachindex(A, B, C)
        @inbounds A[i] += B[i] + s * C[i]
    end
end

function do_norm(A, N)
    asum = 0.0
    for i in 1:N
        @inbounds asum += abs(A[i])
    end
    return asum
end

function (@main)(args)

    MPI.Init()
    comm = MPI.COMM_WORLD
    me = MPI.Comm_rank(comm)
    np = MPI.Comm_size(comm)
    print = (me == 0)

    if (me == 0)
        println("Parallel Research Kernels version")
        println("Julia STREAM triad: A = B + scalar * C")
    end

    if length(args) != 2 && print
        println("argument count = ", length(args))
        println("Usage: mpiexecjl -n N julia --project nstream-mpi.jl <# iterations> <vector length>")
        exit(1)
    end

    argv = map(x->tryparse(Int64,x),args)

    # iterations
    iterations = argv[1]
    if isnothing(iterations) || iterations < 1
        println("ERROR: iterations must be an integer >= 1")
        exit(2)
    end

    # vector length
    vlength = argv[2]
    if isnothing(vlength) || vlength < 1
        println("ERROR: length must be an integer >= 1")
        exit(3)
    end

    if (me == 0)
        println("Number of processes      = ", np)
        println("Number of iterations     = ", iterations)
        println("Vector length            = ", vlength)
    end

    # ********************************************************************
    # ** Allocate space for the input and transpose matrix
    # ********************************************************************

    A = zeros(vlength)
    B = zeros(vlength)
    C = zeros(vlength)
    do_initialize!(A, B, C, vlength)

    scalar = 3.0

    MPI.Barrier(comm)
    t0 = time_ns()

    for _ in 0:iterations
        do_nstream!(A, B, C, scalar, vlength)
    end

    MPI.Barrier(comm)
    t1 = time_ns()
    nstream_time = (t1 - t0) * 1.e-9

    # ********************************************************************
    # ** Analyze and output results.
    # ********************************************************************

    ar = 0.0
    br = 2.0
    cr = 2.0
    for _ in 0:iterations
        ar += br + scalar * cr
    end

    ar *= vlength

    asum = do_norm(A, vlength)

    epsilon = 1.e-8
    if abs(ar-asum)/asum < epsilon
        if (me == 0)
            println("Solution validates")
            avgtime = nstream_time/iterations
            nbytes = 4.0 * np * vlength * sizeof(Float64)
            println("Rate (MB/s): ",1.e-6*nbytes/avgtime, " Avg time (s): ", avgtime)
        end
    else
        if (me == 0)
            println("Failed Validation on output array");
            println("        Expected checksum: ",ar);
            println("        Observed checksum: ",asum);
            println("ERROR: solution did not validate")
        end
        exit(1)
    end
end
