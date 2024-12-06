#!/usr/bin/env python3
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
# HISTORY: - Written by Rob Van der Wijngaart, February 2009.
#          - RvdW: Removed unrolling pragmas for clarity;
#            added constant to array "in" at end of each iteration to force
#            refreshing of neighbor data in parallel versions; August 2013
#          - Converted to Python by Jeff Hammond, February 2016.
#
# *******************************************************************

import sys
print('Python version = ', str(sys.version_info.major)+'.'+str(sys.version_info.minor))
if sys.version_info >= (3, 3):
    from time import process_time as timer
else:
    from timeit import default_timer as timer
import numpy
print('Numpy version  = ', numpy.version.version)

def main():

    # ********************************************************************
    # read and test input parameters
    # ********************************************************************

    print('Parallel Research Kernels')
    print('Python stencil execution on 2D grid')

    if len(sys.argv) < 3:
        print('argument count = ', len(sys.argv))
        sys.exit("Usage: ./stencil <# iterations> <array dimension> [<star/stencil> <radius>]")

    iterations = int(sys.argv[1])
    if iterations < 1:
        sys.exit("ERROR: iterations must be >= 1")

    n = int(sys.argv[2])
    if n < 1:
        sys.exit("ERROR: array dimension must be >= 1")

    if len(sys.argv) > 3:
        pattern = sys.argv[3]
    else:
        pattern = 'star'

    if len(sys.argv) > 4:
        r = int(sys.argv[4])
        if r < 1:
            sys.exit("ERROR: Stencil radius should be positive")
        if (2*r+1) > n:
            sys.exit("ERROR: Stencil radius exceeds grid size")
    else:
        r = 2

    print('Number of iterations = ', iterations)
    print('Grid size            = ', n)
    print('Radius of stencil    = ', r)
    if pattern == 'star':
        print('Type of stencil      = ','star')
    else:
        print('Type of stencil      = ','stencil')
    print('Data type            = double precision')
    print('Compact representation of stencil loop body')

    # there is certainly a more Pythonic way to initialize W,
    # but it will have no impact on performance.
    W = numpy.zeros(((2*r+1),(2*r+1)))
    if pattern == 'star':
        stencil_size = 4*r+1
        for i in range(1,r+1):
            W[r,r+i] = +1./(2*i*r)
            W[r+i,r] = +1./(2*i*r)
            W[r,r-i] = -1./(2*i*r)
            W[r-i,r] = -1./(2*i*r)

    else:
        stencil_size = (2*r+1)**2
        for j in range(1,r+1):
            for i in range(-j+1,j):
                W[r+i,r+j] = +1./(4*j*(2*j-1)*r)
                W[r+i,r-j] = -1./(4*j*(2*j-1)*r)
                W[r+j,r+i] = +1./(4*j*(2*j-1)*r)
                W[r-j,r+i] = -1./(4*j*(2*j-1)*r)

            W[r+j,r+j]    = +1./(4*j*r)
            W[r-j,r-j]    = -1./(4*j*r)

    A = numpy.fromfunction(lambda i,j: i+j, (n,n), dtype=float)
    B = numpy.zeros((n,n))

    for k in range(iterations+1):
        # start timer after a warmup iteration
        if k<1: t0 = timer()

        if pattern == 'star':
            if r==2:
                B[2:n-2,2:n-2] += W[2,2] * A[2:n-2,2:n-2] \
                                + W[2,0] * A[2:n-2,0:n-4] \
                                + W[2,1] * A[2:n-2,1:n-3] \
                                + W[2,3] * A[2:n-2,3:n-1] \
                                + W[2,4] * A[2:n-2,4:n-0] \
                                + W[0,2] * A[0:n-4,2:n-2] \
                                + W[1,2] * A[1:n-3,2:n-2] \
                                + W[3,2] * A[3:n-1,2:n-2] \
                                + W[4,2] * A[4:n-0,2:n-2]
            else:
                b = n-r
                B[r:b,r:b] += W[r,r] * A[r:b,r:b]
                for s in range(1,r+1):
                    B[r:b,r:b] += W[r,r-s] * A[r:b,r-s:b-s] \
                                + W[r,r+s] * A[r:b,r+s:b+s] \
                                + W[r-s,r] * A[r-s:b-s,r:b] \
                                + W[r+s,r] * A[r+s:b+s,r:b]
        else: # stencil
            if r>0:
                b = n-r
                for s in range(-r, r+1):
                    for t in range(-r, r+1):
                        B[r:b,r:b] += W[r+t,r+s] * A[r+t:b+t,r+s:b+s]

        A += 1.0

    t1 = timer()
    stencil_time = t1 - t0

    #******************************************************************************
    #* Analyze and output results.
    #******************************************************************************

    norm = numpy.linalg.norm(numpy.reshape(B,n*n),ord=1)
    active_points = (n-2*r)**2
    norm /= active_points

    epsilon=1.e-8

    # verify correctness
    reference_norm = 2*(iterations+1)
    if abs(norm-reference_norm) < epsilon:
        print('Solution validates')
        flops = (2*stencil_size+1) * active_points
        avgtime = stencil_time/iterations
        print('Rate (MFlops/s): ',1.e-6*flops/avgtime, ' Avg time (s): ',avgtime)
    else:
        print('ERROR: L1 norm = ', norm,' Reference L1 norm = ', reference_norm)
        sys.exit()


if __name__ == '__main__':
    main()
