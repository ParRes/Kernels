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
#
# *******************************************************************

import sys
import time
import numpy

def main():

    # ********************************************************************
    # read and test input parameters
    # ********************************************************************

    print 'Parallel Research Kernels version ' #, PRKVERSION
    print 'Python stencil execution on 2D grid'

    if len(sys.argv) < 3:
        print 'argument count = ', len(sys.argv)
        sys.exit("Usage: ./stencil <# iterations> <array dimension> [<radius> <star/stencil>]")

    iterations = int(sys.argv[1])
    if iterations < 0:
        sys.exit("ERROR: iterations must be >= 1")

    n = int(sys.argv[2])
    if n < 1:
        sys.exit("ERROR: array dimension must be >= 1")

    if len(sys.argv) > 3:
        r = int(sys.argv[3])
        if r < 1:
            sys.exit("ERROR: Stencil radius should be positive")
        if (2*r+1) > n:
            sys.exit("ERROR: Stencil radius exceeds grid size")
    else:
        r = 2 # radius=2 is what other impls use right now

    if len(sys.argv) > 3:
        pattern = sys.argv[4]
    else:
        pattern = 'star'

    print 'Grid size            = ', n
    print 'Radius of stencil    = ', r
    if pattern == 'star':
        print 'Type of stencil      = ','star'
    else:
        print 'Type of stencil      = ','stencil'

    print 'Data type            = double precision'
    print 'Compact representation of stencil loop body'
    print 'Number of iterations = ', iterations

    W = numpy.zeros(((2*r+1),(2*r+1)))
    if pattern == 'star':
        stencil_size = 4*r+1
        for i in range(1,r+1):
            W[r][r+i] = +1./(2*i*r)
            W[r+i][r] = +1./(2*i*r)
            W[r][r-i] = -1./(2*i*r)
            W[r-i][r] = -1./(2*i*r)

    else:
        stencil_size = (2*r+1)**2
        for j in range(1,r+1):
            for i in range(-j+1,j):
                W[r+i][r+j] = +1./(4*j*(2*j-1)*r)
                W[r+i][r-j] = -1./(4*j*(2*j-1)*r)
                W[r+j][r+i] = +1./(4*j*(2*j-1)*r)
                W[r-j][r+i] = -1./(4*j*(2*j-1)*r)

            W[r+j][r+j]    = +1./(4*j*r)
            W[r-j][r-j]    = -1./(4*j*r)

    A = numpy.fromfunction(lambda i,j: i+j, (n,n), dtype=float)
    B = numpy.zeros((n,n))

    for k in range(iterations+1):
        # start timer after a warmup iteration
        if k<1: t0 = time.clock()

        if pattern == 'star':
            if r==2:
                B[2:-2,2:-2] += W[2][2] * A[ 2:-2, 2:-2] \
                              + W[2][0] * A[ 2:-2, 0:-4] \
                              + W[2][1] * A[ 2:-2, 1:-3] \
                              + W[2][3] * A[ 2:-2, 3:-1] \
                              + W[2][4] * A[ 2:-2, 4:  ] \
                              + W[0][2] * A[ 0:-4, 2:-2] \
                              + W[1][2] * A[ 1:-3, 2:-2] \
                              + W[3][2] * A[ 3:-1, 2:-2] \
                              + W[4][2] * A[ 4:  , 2:-2]
            # TODO generalize views impl like this if possible
            #B[r:-r,r:-r] += W[r][r] * A[r:-r,r:-r]
            #for jj in range(-r,0):
            #    B[r:-r,r:-r] += W[r][r+jj] * A[r:-r,r+jj:-r+jj]
            #for jj in range(1,r+1):
            #    B[r:-r,r:-r] += W[r][r+jj] * A[r:-r,r+jj:-r+jj]
            #for ii in range(-r,0):
            #    B[r:-r,r:-r] += W[r+ii][r] * A[ii,r:-r]
            #for ii in range(1,r+1):
            #    B[r:-r,r:-r] += W[r+ii][r] * A[ii,r:-r]
            else: # fallback to bad implementation
                for i in range(r,n-r):
                    for j in range(r,n-r):
                        B[i][j] += W[r][r] * A[i][j]
                        for jj in range(-r,0):
                            B[i][j] += W[r][r+jj] * A[i][j+jj]
                        for jj in range(1,r+1):
                            B[i][j] += W[r][r+jj] * A[i][j+jj]
                        for ii in range(-r,0):
                            B[i][j] += W[r+ii][r] * A[i+ii][j]
                        for ii in range(1,r+1):
                            B[i][j] += W[r+ii][r] * A[i+ii][j]
        else:
            for i in range(r,n-r):
                for j in range(r,n-r):
                    for ii in range(-r,r+1):
                        for jj in range(-r,r+1):
                            B[i][j] += W[r+ii][r+jj] * A[i+ii][j+jj]

        A += 1.0

    t1 = time.clock()
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
        print 'Solution validates'
        flops = (2*stencil_size+1) * active_points
        avgtime = stencil_time/iterations
        print 'Rate (MFlops/s): ',1.e-6*flops/avgtime, ' Avg time (s): ',avgtime
    else:
        print 'ERROR: L1 norm = ', norm,' Reference L1 norm = ', reference_norm
        sys.exit()


if __name__ == '__main__':
    main()

