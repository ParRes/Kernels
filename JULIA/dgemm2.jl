#
# Copyright (c) 2015, Intel Corporation
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
# NAME:    dgemm
#
# PURPOSE: This program tests the efficiency with which a dense matrix
#          dense multiplication is carried out
#
# USAGE:   The program takes as input the matrix order,
#          the number of times the matrix-matrix multiplication
#          is carried out.
#
#          <progname> <# iterations> <matrix order>
#
#          The output consists of diagnostics to make sure the
#          algorithm worked, and of timing statistics.
#
# HISTORY: Written by Rob Van der Wijngaart, February 2009.
#          Converted to Python by Jeff Hammond, February 2016.
#          Fixed timing err, Ave+std_dev, more pythonic, Tim Mattson May 2021
#          Converted to Julia by Carsten Bauer, November 2024.
# *******************************************************************

using LinearAlgebra

function (@main)(args)

    # ********************************************************************
    # read and test input parameters
    # ********************************************************************

    println("Parallel Research Kernels version")
    println("Julia Dense matrix-matrix multiplication: C = A x B")

    if length(args) != 2
        println("argument count = ", length(args))
        println("Usage: julia dgemm.jl <# iterations> <matrix order>")
        exit(1)
    end

    argv = map(x->tryparse(Int64,x),args)

    # iterations
    iterations = argv[1]
    if isnothing(iterations) || iterations < 1
        println("ERROR: iterations must be an integer >= 1")
        exit(2)
    end

    # matrix order
    order = argv[2]
    if isnothing(order) || order < 1
        println("ERROR: length must be an integer >= 1")
        exit(3)
    end

    println("Number of iterations     = ", iterations)
    println("Matrix order             = ", order)

    # ********************************************************************
    # ** Allocate space for the input and transpose matrix
    # ********************************************************************

    A = [float(j) for j in 0:order-1, j in 0:order-1]
    B = copy(A)
    C = zeros(order,order)

    local t0
    t_sum = 0.0
    t_squared = 0.0

    for iter in 0:iterations        # one more warmup iteration
        iter == 1 && (t0 = time_ns()) # start timer in second iteration

        C .+= A * B

        # # Faster alternative:
        # # `mul!(C, A, B, α, β)` computes `A B α + C β` and writes the result into `C`
        # mul!(C, A, B, 1.0, 1.0)

        if iter > 0
             t_iter = time_ns()
             t = t_iter - t0
             t_sum += t
             t_squared = t_squared + t^2
             t0 = t_iter
        end
    end

    dgemm_av  = t_sum / iterations
    dgemm_std = sqrt((t_squared - iterations * dgemm_av * dgemm_av) / (iterations - 1))

    # ********************************************************************
    # ** Analyze and output results.
    # ********************************************************************

    checksum = sum(C)

    ref_checksum = 0.25 * order^3 * (order - 1.0) * (order - 1.0)
    ref_checksum *= (iterations + 1)

    # @show checksum ≈ ref_checksum
    epsilon = 1.e-8
    if abs((checksum - ref_checksum) / ref_checksum) < epsilon
        println("Solution validates")
        nflops = 2.0 * order^3
        recip_diff = (1.0 / (dgemm_av - dgemm_std) - 1.0 / (dgemm_av + dgemm_std))
        gf_std = 1.e-6 * nflops * recip_diff / 2.0
        println("nflops: ", nflops)
        print("Rate: ", 1.e-6 * nflops / dgemm_av, " +/- (MF/s): ", gf_std)
    else
        error("ERROR: Solution did not validate. Checksum = ", checksum,", Reference checksum = ", ref_checksum)
    end
end
