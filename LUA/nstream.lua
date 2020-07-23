#!/usr/bin/env lua

--[[

Copyright (c) 2017, Intel Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

* Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above
     copyright notice, this list of conditions and the following
     disclaimer in the documentation and/or other materials provided
     with the distribution.
* Neither the name of Intel Corporation nor the names of its
     contributors may be used to endorse or promote products
     derived from this software without specific prior written
     permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

******************************************************************

NAME:    nstream

PURPOSE: To compute memory bandwidth when adding a vector of a given
         number of double precision values to the scalar multiple of
         another vector of the same length, and storing the result in
         a third vector.

USAGE:   The program takes as input the number
         of iterations to loop over the triad vectors, the length of the
         vectors, and the offset between vectors

         <progname> <# iterations> <vector length> <offset>

         The output consists of diagnostics to make sure the
         algorithm worked, and of timing statistics.

NOTES:   Bandwidth is determined as the number of words read, plus the
         number of words written, times the size of the words, divided
         by the execution time. For a vector length of N, the total
         number of words read and written is 4*N*sizeof(double).


HISTORY: This code is loosely based on the Stream benchmark by John
         McCalpin, but does not follow all the Stream rules. Hence,
         reported results should not be associated with Stream in
         external publications

         Converted to Python by Jeff Hammond, October 2017.
         Converted to Lua by Jeff Hammond, July 2020.
]]--

-- ********************************************************************
-- read and test input parameters
-- ********************************************************************

print('Parallel Research Kernels')
print('Lua STREAM triad: A = B + scalar * C')

if #arg ~= 2
then
    print('argument count = ', #arg)
    print("Usage: lua nstream.lua <# iterations> <vector length>")
    os.exit()
end

iterations = tonumber(arg[1])
if iterations < 1
then
    print("ERROR: iterations must be >= 1")
    os.exit()
end

length = tonumber(arg[2])
if length < 1
then
    print("ERROR: length must be positive")
    os.exit()
end

print('Number of iterations = ', iterations)
print('Vector length        = ', length)

---------------------------------------------------------
-- Allocate space for the input and execute STREAM triad
---------------------------------------------------------

A = {}
B = {}
C = {}

for i=1,length do
    A[i] = 0.0
    B[i] = 2.0
    C[i] = 2.0
end

scalar = 3.0

for k=0,iterations+1 do

    if k<1
    then
        t0 = os.clock()
    end

    for i=1,length do
        A[i] = A[i] + B[i] + scalar * C[i]
    end

end

t1 = os.clock()
nstream_time = t1 - t0

---------------------------------------------------------
-- Analyze and output results.
---------------------------------------------------------

ar = 0.0
br = 2.0
cr = 2.0
ref = 0.0
for k=0,iterations+1 do
    ar = ar + br + scalar * cr
end

ar = ar * length

asum = 0.0
for i=1,length do
    asum = asum + math.abs(A[i])
end

epsilon=1.e-8
if ( math.abs(ar-asum)/asum > epsilon )
then
    print('Failed Validation on output array')
    print('        Expected checksum: ',ar)
    print('        Observed checksum: ',asum)
    print("ERROR: solution did not validate")
    os.exit()
else
    print('Solution validates')
    avgtime = nstream_time/iterations
    nbytes = 4.0 * length * 8
    print('Rate (MB/s): ',1.e-6*nbytes/avgtime, ' Avg time (s): ', avgtime)
end


