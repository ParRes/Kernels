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
# *******************************************************************

import sys
print('Python version = ', str(sys.version_info.major)+'.'+str(sys.version_info.minor))
if sys.version_info >= (3, 3):
    from time import process_time as timer
else:
    from timeit import default_timer as timer

def main():

    # ********************************************************************
    # read and test input parameters
    # ********************************************************************

    print('Parallel Research Kernels version ') #, PRKVERSION
    print('Python Dense matrix-matrix multiplication: C = A x B')

    if len(sys.argv) != 3:
        print('argument count = ', len(sys.argv))
        sys.exit("Usage: ./transpose <# iterations> <matrix order>")

    iterations = int(sys.argv[1])
    if iterations < 1:
        sys.exit("ERROR: iterations must be >= 1")

    order = int(sys.argv[2])
    if order < 1:
        sys.exit("ERROR: order must be >= 1")

    print('Number of iterations = ', iterations)
    print('Matrix order         = ', order)

    # ********************************************************************
    # ** Allocate space for the input and transpose matrix
    # ********************************************************************

    # 0.0 is a float, which is 64b (53b of precision)
    A = [[0.0 for x in range(order)] for x in range(order)]
    B = [[0.0 for x in range(order)] for x in range(order)]
    C = [[0.0 for x in range(order)] for x in range(order)]

    # this is surely not the Pythonic way of doing this
    for i in range(order):
        for j in range(order):
            A[i][j] = float(j)
            B[i][j] = float(j)

    for k in range(0,iterations+1):

        if k<1: t0 = timer()

        for i in range(order):
            for j in range(order):
                for k in range(order):
                    C[i][j] += A[i][k] * B[k][j]

    t1 = timer()
    dgemm_time = t1 - t0

    # ********************************************************************
    # ** Analyze and output results.
    # ********************************************************************

    checksum = 0.0;
    for i in range(order):
        for j in range(order):
            checksum += C[i][j];

    ref_checksum = 0.25*order*order*order*(order-1.0)*(order-1.0)
    ref_checksum *= (iterations+1)

    epsilon=1.e-8
    if abs((checksum - ref_checksum)/ref_checksum) < epsilon:
        print('Solution validates')
        avgtime = dgemm_time/iterations
        nflops = 2.0*order*order*order
        print('Rate (MF/s): ',1.e-6*nflops/avgtime, ' Avg time (s): ', avgtime)
    else:
        print('ERROR: Checksum = ', checksum,', Reference checksum = ', ref_checksum,'\n')
        sys.exit("ERROR: solution did not validate")


if __name__ == '__main__':
    main()

