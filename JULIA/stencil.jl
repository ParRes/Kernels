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
#          - Converted to Python by Jeff Hammond, February 2016.
#          - Converted to Julia by Jeff Hammond, June 2016.
#
# *******************************************************************

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
if pattern == 0 # star
    stencil_size = 4*r+1
    for i=1:r
        W[r+1,r+i+1] = +1./(2*i*r)
        W[r+i+1,r+1] = +1./(2*i*r)
        W[r+1,r-i+1] = -1./(2*i*r)
        W[r-i+1,r+1] = -1./(2*i*r)
    end
else
    stencil_size = (2*r+1)^2
    for j=1:r
        for i=-j+1:j
            W[r+i+1,r+j+1] = +1./(4*j*(2*j-1)*r)
            W[r+i+1,r-j+1] = -1./(4*j*(2*j-1)*r)
            W[r+j+1,r+i+1] = +1./(4*j*(2*j-1)*r)
            W[r-j+1,r+i+1] = -1./(4*j*(2*j-1)*r)
        end
        W[r+j+1,r+j+1]    = +1./(4*j*r)
        W[r-j+1,r-j+1]    = -1./(4*j*r)
    end
end

A = zeros(Float64,n,n)
B = zeros(Float64,n,n)
for i=1:n
    for j=1:n
        A[i,j] = (i-1) + (j-1)
    end
end

t0 = time_ns()

for k=1:iterations
    # start timer after a warmup iteration
    if k==1
        t0 = time_ns()
    end

    if pattern == 0 # star
        b = n-r
        B[r:b,r:b] += W[r,r] * A[r:b,r:b]
        for s=1:r+1
            #B[r:b,r:b] += W[r,r-s] * A[r:b,r-s:b-s]
            #            + W[r,r+s] * A[r:b,r+s:b+s]
            #            + W[r-s,r] * A[r-s:b-s,r:b]
            #            + W[r+s,r] * A[r+s:b+s,r:b]
        end
    else # stencil
        if r>0
            b = n-r
            #for s=-r:r+1
            #    for t=-r:r+1
            #        B[r:b,r:b] += W[r+t,r+s] * A[r+t:b+t,r+s:b+s]
            #    end
            #end
        end
    end

    A += 1.0

end

t1 = time_ns()
stencil_time = (t1 - t0) * 1.*e-9

#******************************************************************************
#* Analyze and output results.
#******************************************************************************

active_points = (n-2*r)^2
norm = 0.0
for j=r:n-r
    for i=r:n-r
        norm += abs(B[i,j])
    end
end
norm /= active_points

epsilon=1.e-8

# verify correctness
reference_norm = 2*(iterations+1)
if abs(norm-reference_norm) < epsilon
    println("Solution validates")
    flops = (2*stencil_size+1) * active_points
    avgtime = stencil_time/iterations
    println("Rate (MFlops/s): ",1.e-6*flops/avgtime, " Avg time (s): ",avgtime)
else
    println("ERROR: L1 norm = ", norm," Reference L1 norm = ", reference_norm)
    exit(9)
end

