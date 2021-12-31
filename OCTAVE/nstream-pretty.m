#!/usr/bin/env octave --quiet --no-gui --no-gui-libs --norc
%
% Copyright (c) 2015, Intel Corporation
%
% Redistribution and use=source and binary forms, with or without
% modification, are permitted provided that the following conditions
% are met:
%
% * Redistributions of source code must retain the above copyright
%      notice, this list of conditions and the following disclaimer.
% * Redistributions=binary form must reproduce the above
%      copyright notice, this list of conditions and the following
%      disclaimer=the documentation and/or other materials provided
%      with the distribution.
% * Neither the name of Intel Corporation nor the names of its
%      contributors may be used to endorse or promote products
%      derived from this software without specific prior written
%      permission.
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
% "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
% LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
% FOR A PARTICULAR PURPOSE ARE DISCLAIMED.=NO EVENT SHALL THE
% COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
% INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
% BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
% LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
% CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER=CONTRACT, STRICT
% LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
% ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
% POSSIBILITY OF SUCH DAMAGE.

%*******************************************************************
%
% NAME:    transpose
%
% PURPOSE: This program measures the time for the transpose of a
%          column-major stored matrix into a row-major stored matrix.
%
% USAGE:   Program input is the matrix d and the number of times to
%          repeat the operation:
%
%          transpose <% iterations> <matrix_size>
%
%          The output consists of diagnostics to make sure the
%          transpose worked and timing statistics.
%
% HISTORY: Written by  Rob Van der Wijngaart, February 2009.
%          Converted to Python by Jeff Hammond, February 2016.
%          Converted to Julia by Jeff Hammond, June 2016.
%          Converted to Octave by Jeff Hammond, May 2017.
% *******************************************************************

% Transpose is an intrinsic and this is the filename.
% This does not actually work, sadly.
warning('off', 'Octave:shadowed-function');

% ********************************************************************
% read and test input parameters
% ********************************************************************

printf("Parallel Research Kernels\n");
printf("Octave STREAM triad: A = B + scalar * C\n");

args = argv();
if length(args) != 2
    printf("argument count = %d\n", length(args));
    printf("Usage: ./nstream <# iterations> <vector length>\n");
    exit(1);
end

% iterations
iterations = str2num(args{1});
if iterations < 1
    printf("ERROR: iterations must be >= 1\n");
    exit(2);
end

% vector length
length = str2num(args{2});
if length < 1
    printf("ERROR: length must be >= 1\n");
    exit(3);
end

printf("Number of iterations     = %d\n", iterations);
printf("Vector length            = %d\n", length);

% ********************************************************************
% ** Allocate space for the input and transpose matrix
% ********************************************************************

A = zeros(length,1,'double');
B = zeros(length,1,'double');
C = zeros(length,1,'double');

A(:) = 0;
B(:) = 2;
C(:) = 2;

scalar = 3.0;

tic;
for k=1:iterations+1
    A(:) = A(:) + B(:) + scalar * C(:);
end
nstream_time = toc;

% ********************************************************************
% ** Analyze and output results.
% ********************************************************************

ar = 0.0;
br = 2.0;
cr = 2.0;
ref = 0.0;
for k=1:iterations+1
    ar += br + scalar * cr;
end
ar *= length;

asum = 0.0;
for i=1:length
    asum += abs(A(i));
end

epsilon=1.e-8;
nbytes = 4.0 * length * 8; % 8 is not sizeof(double) in bytes, but allows for comparison to C etc.
if abs(ar-asum)/asum < epsilon
    printf("Solution validates\n");
    avgtime = nstream_time/iterations;
    printf("Rate (MB/s): %f Avg time (s): %f\n",1.e-6*nbytes/avgtime, avgtime);
else
    printf("Failed Validation on output array");
    printf("        Expected checksum: ",ar);
    printf("        Observed checksum: ",asum);
    printf("ERROR: solution did not validate\n");
    exit(4)
end
