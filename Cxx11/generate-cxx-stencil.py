#!/usr/bin/env python

import sys
import fileinput
import string
import os

def codegen(src,pattern,stencil_size,radius,W,model):
    if (model=='openmp'):
        src.write('void '+pattern+str(radius)+'(const int n, std::vector<double> & in, std::vector<double> & out) {\n')
        src.write('    _Pragma("omp for")\n')
        src.write('    for (auto i='+str(radius)+'; i<n-'+str(radius)+'; ++i) {\n')
        src.write('      _Pragma("omp simd")\n')
        src.write('      for (auto j='+str(radius)+'; j<n-'+str(radius)+'; ++j) {\n')
    elif (model=='target'):
        src.write('_Pragma("omp declare target")\n')
        src.write('void '+pattern+str(radius)+'(const int n, const double * RESTRICT in, double * RESTRICT out) {\n')
        src.write('    _Pragma("omp for")\n')
        src.write('    for (auto i='+str(radius)+'; i<n-'+str(radius)+'; ++i) {\n')
        src.write('      _Pragma("omp simd")\n')
        src.write('      for (auto j='+str(radius)+'; j<n-'+str(radius)+'; ++j) {\n')
    elif (model=='rangefor'):
        src.write('void '+pattern+str(radius)+'(const int n, std::vector<double> & in, std::vector<double> & out) {\n')
        src.write('    auto inside = boost::irange('+str(radius)+',n-'+str(radius)+');\n')
        src.write('    for (auto i : inside) {\n')
        src.write('      for (auto j : inside) {\n')
    elif (model=='stl'):
        src.write('void '+pattern+str(radius)+'(const int n, std::vector<double> & in, std::vector<double> & out) {\n')
        src.write('    auto inside = boost::irange('+str(radius)+',n-'+str(radius)+');\n')
        src.write('    std::for_each( std::begin(inside), std::end(inside), [&] (int i) {\n')
        src.write('      std::for_each( std::begin(inside), std::end(inside), [&] (int j) {\n')
    elif (model=='pstl'):
        src.write('void '+pattern+str(radius)+'(const int n, std::vector<double> & in, std::vector<double> & out) {\n')
        src.write('    auto inside = boost::irange('+str(radius)+',n-'+str(radius)+');\n')
        src.write('    std::for_each( std::execution::par, std::begin(inside), std::end(inside), [&] (int i) {\n')
        src.write('      std::for_each( std::execution::unseq, std::begin(inside), std::end(inside), [&] (int j) {\n')
    elif (model=='cilk'):
        src.write('void '+pattern+str(radius)+'(const int n, std::vector<double> & in, std::vector<double> & out) {\n')
        src.write('    cilk_for (auto i='+str(radius)+'; i<n-'+str(radius)+'; ++i) {\n')
        src.write('      cilk_for (auto j='+str(radius)+'; j<n-'+str(radius)+'; ++j) {\n')
    elif (model=='kokkos'):
        src.write('void '+pattern+str(radius)+'(const int n, Kokkos::View<double**, Kokkos::LayoutRight> & in, Kokkos::View<double**, Kokkos::LayoutRight> & out) {\n')
        src.write('    Kokkos::parallel_for ( n,[&] (int i) {\n')
        src.write('      for (auto j='+str(radius)+'; j<n-'+str(radius)+'; ++j) {\n')
    elif (model=='tbb'):
        src.write('template <>\n')
        if pattern=='star':
            name='Star'
        elif pattern=='grid':
            name='Grid'
        src.write('struct '+name+'<'+str(radius)+'> {\n')
        src.write('  void operator()( const tbb::blocked_range2d<int>& r ) const {\n')
        src.write('    for (tbb::blocked_range<int>::const_iterator i=r.rows().begin(); i!=r.rows().end(); ++i ) {\n')
        src.write('      for (tbb::blocked_range<int>::const_iterator j=r.cols().begin(); j!=r.cols().end(); ++j ) {\n')
    else:
        src.write('void '+pattern+str(radius)+'(const int n, std::vector<double> & in, std::vector<double> & out) {\n')
        src.write('    for (auto i='+str(radius)+'; i<n-'+str(radius)+'; ++i) {\n')
        src.write('      for (auto j='+str(radius)+'; j<n-'+str(radius)+'; ++j) {\n')
    if (model=='kokkos'):
        src.write('          out(i,j) += ')
    else:
        src.write('          out[i*n+j] += ')
    k = 0
    kmax = stencil_size-1;
    for j in range(0,2*radius+1):
        for i in range(0,2*radius+1):
            if ( W[j][i] != 0.0):
                k+=1
                if (model=='kokkos'):
                    src.write('+in(i+'+str(j-radius)+',j+'+str(i-radius)+') * '+str(W[j][i]))
                else:
                    src.write('+in[(i+'+str(j-radius)+')*n+(j+'+str(i-radius)+')] * '+str(W[j][i]))
                if (k<kmax): src.write('\n')
                if (k>0 and k<kmax): src.write('                      ')
    src.write(';\n')
    if (model=='stl' or model=='pstl' or model=='kokkos'):
        src.write('       }\n')
        src.write('     });\n')
    else:
        src.write('       }\n')
        src.write('     }\n')
    if (model=='tbb'):
        src.write('  }\n\n')
        src.write('    '+name+'(int n, std::vector<double> & in, std::vector<double> & out)\n')
        src.write('        : n(n), in(in), out(out) { }\n\n')
        src.write('    int n;\n')
        src.write('    std::vector<double> & in;\n')
        src.write('    std::vector<double> & out;\n')
        src.write('};\n\n')
    else:
        src.write('}\n\n')

def instance(src,model,pattern,r):

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

    codegen(src,pattern,stencil_size,r,W,model)

def main():
    for model in ['seq','rangefor','stl','pstl','openmp','target','tbb','cilk','kokkos']:
      src = open('stencil_'+model+'.hpp','w')
      src.write('#define RESTRICT __restrict__\n\n')
      for pattern in ['star','grid']:
        for r in range(1,10):
          instance(src,model,pattern,r)
      src.close()

if __name__ == '__main__':
    main()

