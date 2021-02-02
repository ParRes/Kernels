#!/usr/bin/ruby -w
#
#
# Copyright (c) 2020, Intel Corporation
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

###################################################################/
#
# NAME:    nstream
#
# PURPOSE: To compute memory bandwidth when adding a vector of a given
#          number of double precision values to the scalar multiple of
#          another vector of the same length, and storing the result in
#          a third vector.
#
# USAGE:   The program takes as input the number
#          of iterations to loop over the triad vectors and
#          the length of the vectors.
#
#          <progname> <# iterations> <vector length>
#
#          The output consists of diagnostics to make sure the
#          algorithm worked, and of timing statistics.
#
# NOTES:   Bandwidth is determined as the number of words read, plus the
#          number of words written, times the size of the words, divided
#          by the execution time. For a vector length of N, the total
#          number of words read and written is 4*N*sizeof(double).
#
# HISTORY: This code is loosely based on the Stream benchmark by John
#          McCalpin, but does not follow all the Stream rules. Hence,
#          reported results should not be associated with Stream in
#          external publications
#
#          Converted to C# by Jeff Hammond, January 2021.
#
###################################################################/

puts "Parallel Research Kernels"
puts "C# STREAM triad: A = B + scalar * C"

#####################################################################
# Read and test input parameters
#####################################################################

argv = ARGV

if argv.length != 2
    puts 'Usage: <# iterations> <vector length>'
    exit
end

iterations = argv[0].to_i
length     = argv[1].to_i

puts 'Number of iterations = ' + iterations.to_s
puts 'Vector length        = ' + length.to_s

#####################################################################
# Allocate space and perform the computation
#####################################################################

A = Array.new(length)
B = Array.new(length)
C = Array.new(length)

for i in 0..length-1
    A[i] = 0.0
    B[i] = 2.0
    C[i] = 2.0
end

scalar = 3.0;

t0 = Time.now

for k in 0..iterations

    if k == 0
        t0 = Time.now
    end

    for i in 0..length-1
        A[i] += B[i] + scalar * C[i]
    end
end
t1 = Time.now
nstream_time = t1 - t0

#####################################################################
# Analyze and output results
#####################################################################

ar = 0.0
br = 2.0
cr = 2.0
for k in 0..iterations
    ar += br + scalar * cr
end

ar *= length

asum = 0.0;
for i in 0..length-1
    asum += A[i].abs
end

epsilon=1e-8
if ((ar-asum)/asum).abs > epsilon
    puts "Failed Validation on output array"
    puts "       Expected checksum: {0}" + ar.to_s
    puts "       Observed checksum: {0}" + asum.to_s
    puts "ERROR: solution did not validate"
else
    puts "Solution validates"
    avgtime = nstream_time/iterations
    nbytes = 4.0 * length * 8 # use 8=sizeof(double)
    puts 'Rate (MB/s): ' + (1e-6*nbytes/avgtime).to_s + ' Avg time (s): ' + avgtime.to_s
end

