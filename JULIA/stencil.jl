#
# Copyright (c) 2013, Intel Corporation
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
#          - Converted to Python by Jeff Hammond, February 2016.
#          - Converted to Julia by Jeff Hammond, June 2016.
#
# *******************************************************************

function do_add!(A, n)
	for j = axes(A, 2)
		for i = axes(A, 1)
			@inbounds A[i,j] += one(eltype(A))
        end
    end
end

function do_init(A, n)
    for i=1:n
        for j=1:n
            A[i,j] = i+j-2
        end
    end
end

function do_star(A, W, B, r, n)
    for j=r:n-r-1
        for i=r:n-r-1
            for jj=-r:r
                @inbounds B[i+1,j+1] += W[r+1,r+jj+1] * A[i+1,j+jj+1]
            end
            for ii=-r:-1
                @inbounds B[i+1,j+1] += W[r+ii+1,r+1] * A[i+ii+1,j+1]
            end
            for ii=1:r
                @inbounds B[i+1,j+1] += W[r+ii+1,r+1] * A[i+ii+1,j+1]
            end
        end
    end
end

function do_stencil(A, W, B, r, n)
    for j=r:n-r-1
        for i=r:n-r-1
            for jj=-r:r
                for ii=-r:r
                    @inbounds B[i+1,j+1] += W[r+ii+1,r+jj+1] * A[i+ii+1,j+jj+1]
                end
            end
        end
    end
end

function main()
    # ********************************************************************
    # read and test input parameters
    # ********************************************************************

    println("Parallel Research Kernels version ") #, PRKVERSION
    println("Julia stencil execution on 2D grid")

    argc = length(ARGS)
    if argc < 2
        println("argument count = ", length(ARGS))
        println("Usage: ./stencil <# iterations> <array dimension> [<star/stencil> <radius>]")
        exit(1)
    end

    iterations = parse(Int,ARGS[1])
    if iterations < 1
        println("ERROR: iterations must be >= 1")
        exit(2)
    end

    n = parse(Int,ARGS[2])
    if n < 1
        println("ERROR: array dimension must be >= 1")
        exit(3)
    end

    pattern = "star"
    if argc > 2
        pattern = ARGS[3]
    end

    if argc > 3
        r = parse(Int,ARGS[4])
        if r < 1
            println("ERROR: Stencil radius should be positive")
            exit(4)
        elseif (2*r+1) > n
            println("ERROR: Stencil radius exceeds grid size")
            exit(5)
        end
    else
        r = 2 # radius=2 is what other impls use right now
    end

    println("Grid size            = ", n)
    println("Radius of stencil    = ", r)
    if pattern == "star"
        println("Type of stencil      = ","star")
    else
        println("Type of stencil      = ","stencil")
    end

    println("Data type            = double precision")
    println("Compact representation of stencil loop body")
    println("Number of iterations = ", iterations)

    W = zeros(Float64,2*r+1,2*r+1)
    if pattern == "star"
        stencil_size = 4*r+1
        for i=1:r
            W[r+1,r+i+1] =  1.0/(2*i*r)
            W[r+i+1,r+1] =  1.0/(2*i*r)
            W[r+1,r-i+1] = -1.0/(2*i*r)
            W[r-i+1,r+1] = -1.0/(2*i*r)
        end
    else
        stencil_size = (2*r+1)^2
        for j=1:r
            for i=-j+1:j-1
                W[r+i+1,r+j+1] =  1.0/(4*j*(2*j-1)*r)
                W[r+i+1,r-j+1] = -1.0/(4*j*(2*j-1)*r)
                W[r+j+1,r+i+1] =  1.0/(4*j*(2*j-1)*r)
                W[r-j+1,r+i+1] = -1.0/(4*j*(2*j-1)*r)
            end
            W[r+j+1,r+j+1]    =  1.0/(4*j*r)
            W[r-j+1,r-j+1]    = -1.0/(4*j*r)
        end
    end

    precompile(do_init, (Array{Float64,2}, Int64))
    if pattern == "star"
        precompile(do_star, (Array{Float64,2}, Array{Float64,2}, Array{Float64,2}, Int64, Int64))
    else
        precompile(do_stencil, (Array{Float64,2}, Array{Float64,2}, Array{Float64,2}, Int64, Int64))
    end
    precompile(do_add!, (Array{Float64,2}, Int64))

    A = zeros(Float64,n,n)
    B = zeros(Float64,n,n)

    do_init(A, n)

    t0 = time_ns()

    for k in 0:iterations
        if k==0
            t0 = time_ns()
        end
        if pattern == "star"
            do_star(A, W, B, r, n)
        else
            do_stencil(A, W, B, r, n)
        end
        do_add!(A, n)
    end

    t1 = time_ns()
    stencil_time = (t1 - t0) * 1.e-9

    #******************************************************************************
    #* Analyze and output results.
    #******************************************************************************

    active_points = (n-2*r)^2
    actual_norm = 0.0
    for j=1:n
        for i=1:n
            actual_norm += abs(B[i,j])
        end
    end
    actual_norm /= active_points

    epsilon = 1.e-8

    # verify correctness
    reference_norm = 2*(iterations+1)
    if abs(actual_norm-reference_norm) < epsilon
        println("Solution validates")
        flops = (2*stencil_size+1) * active_points
        avgtime = stencil_time/iterations
        println("Rate (MFlops/s): ",1.e-6*flops/avgtime, " Avg time (s): ",avgtime)
    else
        println("ERROR: L1 norm = ", actual_norm, " Reference L1 norm = ", reference_norm)
        exit(9)
    end
end

main()

