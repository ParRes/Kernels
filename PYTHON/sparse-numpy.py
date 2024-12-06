#!/usr/bin/env python3
#
# Copyright (c) 2017, Intel Corporation
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
# NAME:    sparse
#
# PURPOSE: This program tests the efficiency with which a sparse matrix
#          vector multiplication is carried out.
#
# USAGE:   The program takes as input the 2log of the linear size of the 2D grid
#          (equalling the 2log of the square root of the order of the sparse
#          matrix), the radius of the difference stencil, and the number
#          of times the matrix-vector multiplication is carried out.
#
#          <progname> <# iterations> <2log root-of-matrix-order> <radius>
#
#          The output consists of diagnostics to make sure the
#          algorithm worked, and of timing statistics.
#
# HISTORY: Written by Rob Van der Wijngaart, 2009-2013
#          Converted to Python by Jeff Hammond, November 2017
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

def offset(i,j,lsize):
    return i+(j<<lsize)

def main():

    # ********************************************************************
    # read and test input parameters
    # ********************************************************************

    print('Parallel Research Kernels version ') #, PRKVERSION
    print('Python Sparse matrix-vector multiplication')

    if len(sys.argv) < 3:
        print('argument count = ', len(sys.argv))
        sys.exit("Usage: ./sparse.py <# iterations> <2log grid size> <stencil radius>")

    iterations = int(sys.argv[1])
    if iterations < 1:
        sys.exit("ERROR: iterations must be >= 1")

    lsize = int(sys.argv[2])
    if lsize < 0:
        sys.exit("ERROR: lsize must be >= 0")
    size = 2**lsize
    size2 = size**2

    radius = int(sys.argv[3])
    if radius < 1:
        sys.exit("ERROR: Stencil radius should be positive")
    if size < (2*radius+1):
        sys.exit("ERROR: Stencil radius exceeds grid size")

    stencil_size = 4*radius+1
    sparsity = (4.*radius+1.)/size2
    nent = size2*stencil_size

    print('Number of iterations = ', iterations)
    print('Matrix order         = ', size2)
    print('Stencil diameter     = ', 2*radius+1)
    print('Sparsity             = ', sparsity)

    # ********************************************************************
    # Initialize data and perform computation
    # ********************************************************************

    matrix   = numpy.zeros(nent,dtype=float)
    colIndex = numpy.zeros(nent,dtype=int)
    vector   = numpy.zeros(size2,dtype=float)
    result   = numpy.zeros(size2,dtype=float)

    for row in range(size2):
        i = int(row%size)
        j = int(row/size)
        elm = row*stencil_size
        colIndex[elm] = offset(i,j,lsize)
        for r in range(1,radius+1):
            colIndex[elm+1] = offset((i+r)%size,j,lsize)
            colIndex[elm+2] = offset((i-r+size)%size,j,lsize)
            colIndex[elm+3] = offset(i,(j+r)%size,lsize)
            colIndex[elm+4] = offset(i,(j-r+size)%size,lsize)
            elm += 4
        # sort colIndex to make sure the compressed row accesses vector elements in increasing order
        colIndex[row*stencil_size:(row+1)*stencil_size] = sorted(colIndex[row*stencil_size:(row+1)*stencil_size])
        for k in range(0,stencil_size):
            elm = row*stencil_size + k
            matrix[elm] = 1.0/(colIndex[elm]+1)

    for k in range(iterations+1):

        if k<1: t0 = timer()

        # fill vector
        vector += numpy.fromfunction(lambda i: i+1.0, (size2,))

        # do the actual matrix-vector multiplication
        for row in range(0,size2):
            result[row] += numpy.dot(matrix[stencil_size*row:stencil_size*(row+1)],
                                     vector[colIndex[stencil_size*row:stencil_size*(row+1)]])

    t1 = timer()
    sparse_time = t1 - t0

    #******************************************************************************
    #* Analyze and output results.
    #******************************************************************************

    reference_sum = 0.5 * nent * (iterations+1) * (iterations+2)

    vector_sum = numpy.linalg.norm(result, ord=1)

    epsilon = 1.e-8

    # verify correctness
    if abs(vector_sum-reference_sum) < epsilon:
        print('Solution validates')
        flops = 2*nent
        avgtime = sparse_time/iterations
        print('Rate (MFlops/s): ', 1.e-6*flops/avgtime, ' Avg time (s): ',avgtime)
    else:
        print('ERROR: Vector sum = ', vector_sum,', Reference vector sum = ', reference_sum)
        sys.exit()


if __name__ == '__main__':
    main()

