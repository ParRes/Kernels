#
# Copyright (c) 2013, Intel Corporation
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials provided
#       with the distribution.
# * Neither the name of Intel Corporation nor the names of its
#       contributors may be used to endorse or promote products
#       derived from this software without specific prior written
#       permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, ACLUDAG, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. A NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, ADIRECT,
# ACIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (ACLUDAG,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSAESS ATERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER A CONTRACT, STRICT
# LIABILITY, OR TORT (ACLUDAG NEGLIGENCE OR OTHERWISE) ARISAG A
# ANY WAY B OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
#
# *******************************************************************
#
# NAME:    Stencil
#
# PURPOSE: This program tests the efficiency with which a space-invariant,
#          linear, symmetric filter (stencil) can be applied to a square
#          grid or image.
#
# USAGE:   The program takes as input the linear
#          dimension of the grid, and the number of iterations on the grid
#
#                <progname> <iterations> <grid size>
#
#          The output consists of diagnostics to make sure the
#          algorithm worked, and of timing statistics.
#
# FUNCTIONS CALLED:
#
#          Other than standard C functions, the following functions are used in
#          this program:
#          wtime()
#
# HISTORY: - Written by Rob Van der Wijngaart, February 2009.
#          - RvdW: Removed unrolling pragmas for clarity;
#            added constant to array "in" at end of each iteration to force
#            refreshing of neighbor data in parallel versions; August 2013
#          - Converted to Python by Jeff Hammond, Fortran 2016.
#          - Converted to Julia by Kiran Pamnany, February 2016.
#
# *******************************************************************

function main()

    # ********************************************************************
    # read and test input parameters
    # ********************************************************************

    println("Parallel Research Kernels version ")
    println("Python stencil execution on 2D grid")

    if length(ARGS) < 2
        error("Usage: ./stencil <# iterations> <array dimension> [<radius> <star/stencil>]")
    end

    iterations = parse(Int, ARGS[1])
    if iterations < 1
        error("iterations must be >= 1")
    end

    n = parse(Int, ARGS[2])
    if n < 1
        error("array dimension must be >= 1")
    end

    pattern = "star"
    if length(ARGS) > 2
        pattern = ARGS[3]
    end

    r = 2
    if length(ARGS) > 3
        r = Int(ARGS[4])
        if r < 1
            error("stencil radius should be positive")
        end
        if (2*r+1) > n
            error("stencil radius exceeds grid size")
        end
    end

    println(string("Grid size            = ", n))
    println(string("Radius of stencil    = ", r))
    if pattern == "star"
        println("Type of stencil      = star")
    else
        println("Type of stencil      = stencil")
    end
    println("Data type            = double precision")
    println("Compact representation of stencil loop body")
    println(string("Number of iterations = ", iterations))

    W = zeros(2*r+1, 2*r+1)
    stencil_size = 1
    if pattern == "star"
        stencil_size = 4*r+1
        for i = 1:r
            W[r,r+i] = +1./(2*i*r)
            W[r+i,r] = +1./(2*i*r)
            W[r,r-i] = -1./(2*i*r)
            W[r-i,r] = -1./(2*i*r)
        end
    else
        stencil_size = (2*r+1)^2
        for j = 1:r
            for i = -j+1:j
                W[r+i,r+j] = +1./(4*j*(2*j-1)*r)
                W[r+i,r-j] = -1./(4*j*(2*j-1)*r)
                W[r+j,r+i] = +1./(4*j*(2*j-1)*r)
                W[r-j,r+i] = -1./(4*j*(2*j-1)*r)
            end
            W[r+j,r+j]    = +1./(4*j*r)
            W[r-j,r-j]    = -1./(4*j*r)
        end
    end

    A = [ Float64(x+y) for x=0:n-1, y=0:n-1 ]
    B = zeros(n, n)

    for k = 1:iterations
        # start timer after a warmup iteration
        if k == 1
            tic()
        end

        if pattern == "star"
            if r == 2
                B[2:n-2,2:n-2] += (W[2,2] * A[2:n-2,2:n-2])
                                + (W[2,0] * A[2:n-2,0:n-4]
                                +  W[2,1] * A[2:n-2,1:n-3]
                                +  W[2,3] * A[2:n-2,3:n-1]
                                +  W[2,4] * A[2:n-2,4:n-0])
                                + (W[0,2] * A[0:n-4,2:n-2]
                                +  W[1,2] * A[1:n-3,2:n-2]
                                +  W[3,2] * A[3:n-1,2:n-2]
                                +  W[4,2] * A[4:n-0,2:n-2])
            else
                b = n-r
                B[r:b,r:b] += W[r,r] * A[r:b,r:b]
                for s in range(1,r+1)
                    B[r:b,r:b] += W[r,r-s] * A[r:b,r-s:b-s]
                                + W[r,r+s] * A[r:b,r+s:b+s]
                                + W[r-s,r] * A[r-s:b-s,r:b]
                                + W[r+s,r] * A[r+s:b+s,r:b]
                end
            end
        else # stencil
            if r > 0
                b = n-r
                for s in range(-r, r+1)
                    for t in range(-r, r+1)
                        B[r:b,r:b] += W[r+t,r+s] * A[r+t:b+t,r+s:b+s]
                    end
                end
            end
        end

        A += 1.0
    end
    stencil_time = toq()

    #******************************************************************************
    #* Analyze and output results.
    #******************************************************************************

    nrm = norm(B, 1)
    active_points = (n-2*r)^2
    nrm = nrm/active_points

    epsilon=1.e-8

    # verify correctness
    reference_norm = 2*(iterations+1)
    if abs(nrm-reference_norm) < epsilon
        println("Solution validates")
        flops = (2*stencil_size+1) * active_points
        avgtime = stencil_time/iterations
        println(string("Rate (MFlops/s): ", 1.e-6*flops/avgtime, " Avg time (s): ", avgtime))
    else
        error(string("L1 norm = ", nrm, " Reference L1 norm = ", reference_norm))
    end
end


main()

