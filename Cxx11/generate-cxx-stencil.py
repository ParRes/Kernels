#!/usr/bin/env python

import sys
import fileinput
import string
import os

def codegen(pattern,stencil_size,radius,W,model):
    src = open(pattern+str(radius)+'_'+model+'.h','a')
    src.write('#define RESTRICT __restrict__\n\n')
    if (model=='openmp'):
        src.write('void '+pattern+str(radius)+'(const int n, const double * RESTRICT in, double * RESTRICT out) {\n')
        src.write('_Pragma("omp for")\n')
        src.write('    for (auto i='+str(radius)+'; i<n-'+str(radius)+'; ++i) {\n')
        src.write('_Pragma("omp simd)\n')
        src.write('      for (auto j='+str(radius)+'; j<n-'+str(radius)+'; ++j) {\n')
    elif (model=='rangefor'):
        src.write('void '+pattern+str(radius)+'(const int n, const double * RESTRICT in, double * RESTRICT out) {\n')
        src.write('    auto inside = boost::irange('+str(radius)+',n-'+str(radius)+');\n')
        src.write('    for (auto i : inside) {\n')
        src.write('      for (auto j : inside) {\n')
    elif (model=='stl'):
        src.write('void '+pattern+str(radius)+'(const int n, const double * RESTRICT in, double * RESTRICT out) {\n')
        src.write('    auto inside = boost::irange('+str(radius)+',n-'+str(radius)+');\n')
        src.write('    std::for_each( std::begin(inside), std::end(inside), [&] (int i) {\n')
        src.write('      std::for_each( std::begin(inside), std::end(inside), [&] (int j) {\n')
    elif (model=='pstl'):
        src.write('void '+pattern+str(radius)+'(const int n, const double * RESTRICT in, double * RESTRICT out) {\n')
        src.write('    auto inside = boost::irange('+str(radius)+',n-'+str(radius)+');\n')
        src.write('    std::for_each( std::execution::par, std::begin(inside), std::end(inside), [&] (int i) {\n')
        src.write('      std::for_each( std::execution::unseq, std::begin(inside), std::end(inside), [&] (int j) {\n')
    elif (model=='cilk'):
        src.write('void '+pattern+str(radius)+'(const int n, const double * RESTRICT in, double * RESTRICT out) {\n')
        src.write('    cilk_for (auto i='+str(radius)+'; i<n-'+str(radius)+'; ++i) {\n')
        src.write('      cilk_for (auto j='+str(radius)+'; j<n-'+str(radius)+'; ++j) {\n')
    elif (model=='tbb'):
        src.write('void operator()( const tbb::blocked_range2d<int>& r ) const {\n')
        src.write('    for (tbb::blocked_range<int>::const_iterator i=r.rows().begin(); i!=r.rows().end(); ++i ) {\n')
        src.write('      for (tbb::blocked_range<int>::const_iterator j=r.cols().begin(); j!=r.cols().end(); ++j ) {\n')
    else:
        src.write('void '+pattern+str(radius)+'(const int n, const double * RESTRICT in, double * RESTRICT out) {\n')
        src.write('    for (auto i='+str(radius)+'; i<n-'+str(radius)+'; ++i) {\n')
        src.write('      for (auto j='+str(radius)+'; j<n-'+str(radius)+'; ++j) {\n')
    src.write('        out[i*n+j] += ')
    k = 0
    kmax = stencil_size-1;
    for j in range(0,2*radius+1):
        for i in range(0,2*radius+1):
            if ( W[j][i] != 0.0):
                k+=1
                src.write('+in[(i+'+str(j-radius)+')*n+(j+'+str(i-radius)+')] * '+str(W[j][i]))
                if (k<kmax): src.write('\n')
                if (k>0 and k<kmax): src.write('                      ')
    src.write(';\n')
    src.write('       }\n')
    src.write('     }\n')
    src.write('}\n\n')
    src.close()

def instance(pattern,r):

    if len(sys.argv) < 3:
        print('argument count = ', len(sys.argv))
        sys.exit("Usage: <star/grid> <radius>")

    if len(sys.argv) > 1:
        pattern = sys.argv[1]
    else:
        pattern = 'star'

    if len(sys.argv) > 2:
        r = int(sys.argv[2])
        if r < 1:
            sys.exit("ERROR: Stencil radius should be positive")
    else:
        r = 2 # radius=2 is what other impls use right now

    #if pattern == 'star':
    #    print('Type of stencil      = ', 'star')
    #else:
    #    print('Type of stencil      = ', 'stencil')
    #print('Radius of stencil    = ', r)

    W = [[0.0 for x in range(2*r+1)] for x in range(2*r+1)]
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

    codegen(pattern,stencil_size,r,W,'seq')
    codegen(pattern,stencil_size,r,W,'rangefor')
    codegen(pattern,stencil_size,r,W,'stl')
    codegen(pattern,stencil_size,r,W,'pstl')
    codegen(pattern,stencil_size,r,W,'openmp')
    codegen(pattern,stencil_size,r,W,'tbb')
    codegen(pattern,stencil_size,r,W,'cilk')

def main():

    if len(sys.argv) < 3:
        print('argument count = ', len(sys.argv))
        sys.exit("Usage: <star/grid> <radius>")

    if len(sys.argv) > 1:
        pattern = sys.argv[1]
    else:
        pattern = 'star'

    if len(sys.argv) > 2:
        r = int(sys.argv[2])
        if r < 1:
            sys.exit("ERROR: Stencil radius should be positive")
    else:
        r = 2 # radius=2 is what other impls use right now

    instance(pattern,r)

if __name__ == '__main__':
    main()

