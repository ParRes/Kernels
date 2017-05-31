#!/usr/bin/env octave --quiet --no-gui --no-gui-libs --norc
%
% Copyright (c) 2013, Intel Corporation
%
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions
% are met:
%
% * Redistributions of source code must retain the above copyright
%       notice, this list of conditions and the following disclaimer.
% * Redistributions in binary form must reproduce the above
%       copyright notice, this list of conditions and the following
%       disclaimer in the documentation and/or other materials provided
%       with the distribution.
% * Neither the name of Intel Corporation nor the names of its
%       contributors may be used to endorse or promote products
%       derived from this software without specific prior written
%       permission.
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
% "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, ACLUDAG, BUT NOT
% LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
% FOR A PARTICULAR PURPOSE ARE DISCLAIMED. A NO EVENT SHALL THE
% COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, ADIRECT,
% ACIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (ACLUDAG,
% BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
% LOSS OF USE, DATA, OR PROFITS; OR BUSAESS ATERRUPTION) HOWEVER
% CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER A CONTRACT, STRICT
% LIABILITY, OR TORT (ACLUDAG NEGLIGENCE OR OTHERWISE) ARISAG A
% ANY WAY B OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
% POSSIBILITY OF SUCH DAMAGE.
%
%
% *******************************************************************
%
% NAME:    Stencil
%
% PURPOSE: This program tests the efficiency with which a space-invariant,
%          linear, symmetric filter (stencil) can be applied to a square
%          grid or image.
%
% USAGE:   The program takes as input the linear
%          dimension of the grid, and the number of iterations on the grid
%
%                <progname> <iterations> <grid size>
%
%          The output consists of diagnostics to make sure the
%          algorithm worked, and of timing statistics.
%
% FUNCTIONS CALLED:
%
%          Other than standard C functions, the following functions are used in
%          this program:
%          wtime()
%
% HISTORY: - Written by Rob Van der Wijngaart, February 2009.
%          - RvdW: Removed unrolling pragmas for clarity;
%            added constant to array "in" at end of each iteration to force
%            refreshing of neighbor data in parallel versions; August 2013
%          - Converted to Python by Jeff Hammond, February 2016.
%          - Converted to Julia by Jeff Hammond, June 2016.
%          - Converted to Octave by Jeff Hammond, May 2017.
%
% *******************************************************************

% ********************************************************************
% read and test input parameters
% ********************************************************************

printf("Parallel Research Kernels version\n");
printf("Octave stencil execution on 2D grid\n");

args = argv();
argc = length(args);
if argc < 2
    printf("argument count = %d\n", argc);
    printf("Usage: ./stencil <% iterations> <array dimension> (<star/grid> <radius>)\n");
    exit(1);
end

% iterations
iterations = str2num(args{1});
if iterations < 1
    printf("ERROR: iterations must be >= 1\n");
    exit(2);
end

% grid dimension
n = str2num(args{2}); 
if n < 1
    printf("ERROR: array dimension must be >= 1\n");
    exit(3);
end

% stencil pattern
pattern = "star";
if argc > 2
    pattern = args{3};
end

% stencil radius
if argc > 3
    r = str2num(args{4});
    if r < 1
        printf("ERROR: Stencil radius should be positive\n");
        exit(4);
    elseif (2*r+1) > n
        printf("ERROR: Stencil radius exceeds grid size\n");
        exit(5);
    end
else
    r = 2; % radius=2 is what other impls use right now
end

printf("Grid size            = %d\n", n);
printf("Radius of stencil    = %d\n", r);
if pattern == "star";
    printf("Type of stencil      = %s\n","star");
else
    printf("Type of stencil      = %s\n","grid");
end

printf("Data type            = double precision\n");
printf("Compact representation of stencil loop body\n");
printf("Number of iterations = %d\n", iterations);

W = zeros(2*r+1,2*r+1,'double');
if pattern == "star";
    stencil_size = 4*r+1;
    for i=1:r
        W(r+1,r+i+1) = +1./(2*i*r);
        W(r+i+1,r+1) = +1./(2*i*r);
        W(r+1,r-i+1) = -1./(2*i*r);
        W(r-i+1,r+1) = -1./(2*i*r);
    end
else
    stencil_size = (2*r+1)^2;
    for j=1:r
        for i=-j+1:j-1
            W(r+i+1,r+j+1) = +1./(4*j*(2*j-1)*r);
            W(r+i+1,r-j+1) = -1./(4*j*(2*j-1)*r);
            W(r+j+1,r+i+1) = +1./(4*j*(2*j-1)*r);
            W(r-j+1,r+i+1) = -1./(4*j*(2*j-1)*r);
        end
        W(r+j+1,r+j+1)    = +1./(4*j*r);
        W(r-j+1,r-j+1)    = -1./(4*j*r);
    end
end

A = zeros(n,n,'double');
for i=1:n
    for j=1:n
        A(i,j) = i+j-2;
    end
end
B = zeros(n,n,'double');

tic;
for k=1:iterations+1
    if pattern == "star"
        for j=r:n-r-1
            for i=r:n-r-1
                B(i+1,j+1) += W(r+1,r+1) * A(i+1,j+1);
                for s=1:r
                    B(i+1,j+1) += W(r+1,r-s+1) * A(i+1,j-s+1);
                    B(i+1,j+1) += W(r+1,r+s+1) * A(i+1,j+s+1);
                    B(i+1,j+1) += W(r-s+1,r+1) * A(i-s+1,j+1);
                    B(i+1,j+1) += W(r+s+1,r+1) * A(i+s+1,j+1);
                end
            end
        end
    else
        for j=r:n-r-1
            for i=r:n-r-1
                for jj=-r:r
                    for ii=-r:r
                        B(i+1,j+1) += W(r+ii+1,r+jj+1) * A(i+ii+1,j+jj+1);
                    end
                end
            end
        end
    end
    A += 1.0;
end
stencil_time = toc;

%******************************************************************************
%* Analyze and output results.
%******************************************************************************

active_points = (n-2*r)^2;
actual_norm = 0.0;
for j=1:n
    for i=1:n
        actual_norm += abs(B(i,j));
    end
end
actual_norm /= active_points;

epsilon=1.e-8;

% verify correctness
reference_norm = 2*(iterations+1);
if abs(actual_norm-reference_norm) < epsilon
    printf("Solution validates\n");
    flops = (2*stencil_size+1) * active_points;
    avgtime = stencil_time/iterations;
    printf("Rate (MFlops/s): %f Avg time (s): %f\n", 1.e-6*flops/avgtime, avgtime);
else
    printf("ERROR: L1 norm = %f Reference L1 norm = %f\n", actual_norm, reference_norm);
    exit(9);
end
