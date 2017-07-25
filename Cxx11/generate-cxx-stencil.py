#!/usr/bin/env python3

import sys
import fileinput
import string
import os

def codegen(src,pattern,stencil_size,radius,W,model):
    if (model=='openmp'):
        src.write('void '+pattern+str(radius)+'(const int n, std::vector<double> & in, std::vector<double> & out) {\n')
        src.write('    OMP_FOR()\n')
        src.write('    for (auto i='+str(radius)+'; i<n-'+str(radius)+'; ++i) {\n')
        src.write('      OMP_SIMD\n')
        src.write('      for (auto j='+str(radius)+'; j<n-'+str(radius)+'; ++j) {\n')
    elif (model=='taskloop'):
        src.write('void '+pattern+str(radius)+'(const int n, std::vector<double> & in, std::vector<double> & out) {\n')
        src.write('    OMP_TASKLOOP( firstprivate(n) shared(in,out) )\n')
        src.write('    for (auto i='+str(radius)+'; i<n-'+str(radius)+'; ++i) {\n')
        src.write('      OMP_SIMD\n')
        src.write('      for (auto j='+str(radius)+'; j<n-'+str(radius)+'; ++j) {\n')
    elif (model=='target'):
        src.write('void '+pattern+str(radius)+'(const int n, const double * RESTRICT in, double * RESTRICT out) {\n')
        src.write('    OMP_FOR()\n')
        src.write('    for (auto i='+str(radius)+'; i<n-'+str(radius)+'; ++i) {\n')
        src.write('      OMP_SIMD\n')
        src.write('      for (auto j='+str(radius)+'; j<n-'+str(radius)+'; ++j) {\n')
    elif (model=='rangefor'):
        src.write('void '+pattern+str(radius)+'(const int n, std::vector<double> & in, std::vector<double> & out) {\n')
        src.write('    auto inside = boost::irange('+str(radius)+',n-'+str(radius)+');\n')
        src.write('    for (auto i : inside) {\n')
        src.write('      PRAGMA_SIMD\n')
        src.write('      for (auto j : inside) {\n')
    elif (model=='stl'):
        src.write('void '+pattern+str(radius)+'(const int n, std::vector<double> & in, std::vector<double> & out) {\n')
        src.write('    auto inside = boost::irange('+str(radius)+',n-'+str(radius)+');\n')
        src.write('    std::for_each( std::begin(inside), std::end(inside), [&] (int i) {\n')
        src.write('      PRAGMA_SIMD\n')
        src.write('      std::for_each( std::begin(inside), std::end(inside), [&] (int j) {\n')
    elif (model=='pgnu'):
        src.write('void '+pattern+str(radius)+'(const int n, std::vector<double> & in, std::vector<double> & out) {\n')
        src.write('    auto inside = boost::irange('+str(radius)+',n-'+str(radius)+');\n')
        src.write('    __gnu_parallel::for_each( std::begin(inside), std::end(inside), [&] (int i) {\n')
        src.write('      std::for_each( std::begin(inside), std::end(inside), [&] (int j) {\n')
    elif (model=='pstl'):
        src.write('void '+pattern+str(radius)+'(const int n, std::vector<double> & in, std::vector<double> & out) {\n')
        src.write('    auto inside = boost::irange('+str(radius)+',n-'+str(radius)+');\n')
        src.write('    std::for_each( std::execution::par, std::begin(inside), std::end(inside), [&] (int i) {\n')
        src.write('      std::for_each( std::execution::unseq, std::begin(inside), std::end(inside), [&] (int j) {\n')
    elif (model=='cilk'):
        src.write('void '+pattern+str(radius)+'(const int n, std::vector<double> & in, std::vector<double> & out) {\n')
        src.write('    _Cilk_for (auto i='+str(radius)+'; i<n-'+str(radius)+'; ++i) {\n')
        src.write('      PRAGMA_SIMD\n')
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
    elif (model=='raja'):
        src.write('void '+pattern+str(radius)+'(const int n, std::vector<double> & in, std::vector<double> & out) {\n')
        src.write('    RAJA::forallN<RAJA::NestedPolicy<RAJA::ExecList<thread_exec, RAJA::simd_exec>>>\n')
        src.write('            ( RAJA::RangeSegment('+str(radius)+',n-'+str(radius)+'),'
                                'RAJA::RangeSegment('+str(radius)+',n-'+str(radius)+'),\n')
        src.write('              [&](RAJA::Index_type i, RAJA::Index_type j) {\n')
    elif (model=='kokkos'):
        src.write('void '+pattern+str(radius)+'(const int n, matrix & in, matrix & out) {\n')
        src.write('    Kokkos::parallel_for ( Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>('+str(radius)+',n-'+str(radius)+'), KOKKOS_LAMBDA(const int i) {\n')
        src.write('      for (auto j='+str(radius)+'; j<n-'+str(radius)+'; ++j) {\n')
    else:
        src.write('void '+pattern+str(radius)+'(const int n, std::vector<double> & in, std::vector<double> & out) {\n')
        src.write('    for (auto i='+str(radius)+'; i<n-'+str(radius)+'; ++i) {\n')
        src.write('      PRAGMA_SIMD\n')
        src.write('      for (auto j='+str(radius)+'; j<n-'+str(radius)+'; ++j) {\n')
    if (model=='kokkos'):
        src.write('        out(i,j) += ')
    else:
        src.write('        out[i*n+j] += ')
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
    if (model=='stl' or model=='pgnu' or model=='pstl'):
        src.write('       });\n')
        src.write('     });\n')
    elif (model=='kokkos'):
        src.write('       }\n')
        src.write('     });\n')
    elif (model=='raja'):
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

    W = [[0.0e0 for x in range(2*r+1)] for x in range(2*r+1)]
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
    for model in ['seq','rangefor','stl','pgnu','pstl','openmp','taskloop','target','tbb','cilk','raja','kokkos']:
      src = open('stencil_'+model+'.hpp','w')
      src.write('#define RESTRICT __restrict__\n\n')
      if (model=='target'):
          src.write('OMP_DECLARE_TARGET\n')
      if (model=='raja'):
          src.write('#ifdef RAJA_ENABLE_OPENMP\n')
          src.write('  typedef RAJA::omp_parallel_for_exec thread_exec;\n')
          src.write('#else\n')
          src.write('  typedef RAJA::seq_exec thread_exec;\n')
          src.write('#endif\n')
      for pattern in ['star','grid']:
        for r in range(1,10):
          instance(src,model,pattern,r)
      if (model=='target'):
          src.write('OMP_END_DECLARE_TARGET\n')
      src.close()

if __name__ == '__main__':
    main()

