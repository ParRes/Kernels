#!/usr/bin/env python

import sys
import fileinput
import string
import os

def bodygen(src,pattern,stencil_size,radius,W,model):
    if (model=='kokkos' or model=='rajaview'):
        src.write('              out(i,j) += ')
    else:
        src.write('            out[i*n+j] += ')
    k = 0
    kmax = stencil_size-1;
    for j in range(0,2*radius+1):
        if (j-radius)<0:
            jr=str(j-radius)
        elif (j-radius)==0:
            jr=''
        else:
            jr='+'+str(j-radius)

        for i in range(0,2*radius+1):
            if (i-radius)<0:
                ir=str(i-radius)
            elif (i-radius)==0:
                ir=''
            else:
                ir='+'+str(i-radius)

            if ( W[j][i] != 0.0):
                k+=1
                if (model=='kokkos' or model=='rajaview'):
                    src.write('+in(i'+ir+',j'+jr+') * '+str(W[j][i]))
                else:
                    src.write('+in[(i'+ir+')*n+(j'+jr+')] * '+str(W[j][i]))
                if (k<kmax): src.write('\n')
                if (k>0 and k<kmax): src.write('                          ')
    src.write(';\n')

def codegen(src,pattern,stencil_size,radius,W,model):
    if (model=='openmp'):
        src.write('void '+pattern+str(radius)+'(const int n, const int t, const double * RESTRICT in, double * RESTRICT out) {\n')
        src.write('    OMP_FOR(collapse(2))\n')
        src.write('    for (auto it='+str(radius)+'; it<n-'+str(radius)+'; it+=t) {\n')
        src.write('      for (auto jt='+str(radius)+'; jt<n-'+str(radius)+'; jt+=t) {\n')
        src.write('        for (auto i=it; i<std::min(n-'+str(radius)+',it+t); ++i) {\n')
        src.write('          OMP_SIMD\n')
        src.write('          for (auto j=jt; j<std::min(n-'+str(radius)+',jt+t); ++j) {\n')
        bodygen(src,pattern,stencil_size,radius,W,model)
        src.write('           }\n')
        src.write('         }\n')
        src.write('       }\n')
        src.write('     }\n')
    elif (model=='taskloop'):
        src.write('void '+pattern+str(radius)+'(const int n, const int t, prk::vector<double> & in, prk::vector<double> & out, const int gs) {\n')
        src.write('    OMP_TASKLOOP_COLLAPSE(2, firstprivate(n) shared(in,out) grainsize(gs) )\n')
        src.write('    for (auto it='+str(radius)+'; it<n-'+str(radius)+'; it+=t) {\n')
        src.write('      for (auto jt='+str(radius)+'; jt<n-'+str(radius)+'; jt+=t) {\n')
        src.write('        for (auto i=it; i<std::min(n-'+str(radius)+',it+t); ++i) {\n')
        src.write('          OMP_SIMD\n')
        src.write('          for (auto j=jt; j<std::min(n-'+str(radius)+',jt+t); ++j) {\n')
        bodygen(src,pattern,stencil_size,radius,W,model)
        src.write('           }\n')
        src.write('         }\n')
        src.write('       }\n')
        src.write('     }\n')
    elif (model=='target'):
        src.write('void '+pattern+str(radius)+'(const int n, const int t, const double * RESTRICT in, double * RESTRICT out) {\n')
        src.write('    OMP_TARGET( teams distribute parallel for simd collapse(2) schedule(static,1) )\n')
        src.write('    for (auto i='+str(radius)+'; i<n-'+str(radius)+'; ++i) {\n')
        src.write('      for (auto j='+str(radius)+'; j<n-'+str(radius)+'; ++j) {\n')
        bodygen(src,pattern,stencil_size,radius,W,model)
        src.write('       }\n')
        src.write('     }\n')
    elif (model=='rangefor'):
        src.write('void '+pattern+str(radius)+'(const int n, const int t, prk::vector<double> & in, prk::vector<double> & out) {\n')
        src.write('    auto inside = prk::range('+str(radius)+',n-'+str(radius)+');\n')
        src.write('    for (auto i : inside) {\n')
        src.write('      PRAGMA_SIMD\n')
        src.write('      for (auto j : inside) {\n')
        bodygen(src,pattern,stencil_size,radius,W,model)
        src.write('           }\n')
        src.write('         }\n')
        src.write('       }\n')
        src.write('     }\n')
    elif (model=='stl'):
        src.write('void '+pattern+str(radius)+'(const int n, const int t, std::vector<double> & in, std::vector<double> & out) {\n')
        src.write('    auto inside = prk::range('+str(radius)+',n-'+str(radius)+');\n')
        src.write('    std::for_each( std::begin(inside), std::end(inside), [&] (int i) {\n')
        #src.write('      PRAGMA_SIMD\n')
        src.write('      std::for_each( std::begin(inside), std::end(inside), [&] (int j) {\n')
        bodygen(src,pattern,stencil_size,radius,W,model)
        src.write('      });\n')
        src.write('    });\n')
    elif (model=='pgnu'):
        src.write('void '+pattern+str(radius)+'(const int n, const int t, std::vector<double> & in, std::vector<double> & out) {\n')
        src.write('    auto inside = prk::range('+str(radius)+',n-'+str(radius)+');\n')
        src.write('    __gnu_parallel::for_each( std::begin(inside), std::end(inside), [&] (int i) {\n')
        src.write('      std::for_each( std::begin(inside), std::end(inside), [&] (int j) {\n')
        bodygen(src,pattern,stencil_size,radius,W,model)
        src.write('      });\n')
        src.write('    });\n')
    elif (model=='pstl'):
        src.write('void '+pattern+str(radius)+'(const int n, const int t, std::vector<double> & in, std::vector<double> & out) {\n')
        src.write('    auto inside = prk::range('+str(radius)+',n-'+str(radius)+');\n')
        src.write('    std::for_each( exec::par, std::begin(inside), std::end(inside), [&] (int i) {\n')
        src.write('      std::for_each( exec::unseq, std::begin(inside), std::end(inside), [&] (int j) {\n')
        bodygen(src,pattern,stencil_size,radius,W,model)
        src.write('      });\n')
        src.write('    });\n')
    elif (model=='raja'):
        src.write('void '+pattern+str(radius)+'(const int n, const int t, std::vector<double> & in, std::vector<double> & out) {\n')
        src.write('    RAJA::RangeSegment inside('+str(radius)+',n-'+str(radius)+');\n')
        src.write('    RAJA::forall<thread_exec>(inside, [&](RAJA::Index_type i) {\n')
        src.write('      RAJA::forall<RAJA::simd_exec>(inside, [&](RAJA::Index_type j) {\n')
        bodygen(src,pattern,stencil_size,radius,W,model)
        src.write('      });\n')
        src.write('    });\n')
    elif (model=='rajaview'):
        src.write('void '+pattern+str(radius)+'(const int n, const int t, matrix & in, matrix & out) {\n')
        src.write('    RAJA::RangeSegment inner1('+str(radius)+',n-'+str(radius)+');\n')
        src.write('    auto inner2 = RAJA::make_tuple(inner1, inner1);\n')
        src.write('    RAJA::kernel<regular_policy>(inner2, [=](int i, int j) {\n')
        bodygen(src,pattern,stencil_size,radius,W,model)
        src.write('    });\n')
    elif (model=='tbb'):
        src.write('void '+pattern+str(radius)+'(const int n, const int t, prk::vector<double> & in, prk::vector<double> & out) {\n')
        src.write('  tbb::blocked_range2d<int> range('+str(radius)+', n-'+str(radius)+', t, '+str(radius)+', n-'+str(radius)+', t);\n')
        src.write('  tbb::parallel_for( range, [&](decltype(range)& r ) {\n')
        src.write('    for (auto i=r.rows().begin(); i!=r.rows().end(); ++i ) {\n')
        src.write('      PRAGMA_SIMD\n')
        src.write('      for (auto j=r.cols().begin(); j!=r.cols().end(); ++j ) {\n')
        bodygen(src,pattern,stencil_size,radius,W,model)
        src.write('      }\n')
        src.write('    }\n')
        src.write('  }, tbb_partitioner );\n')
    elif (model=='kokkos'):
        src.write('void '+pattern+str(radius)+'(const int n, const int t, matrix & in, matrix & out) {\n')
        src.write('    auto inside = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({'+str(radius)+','+str(radius)+'},{n-'+str(radius)+',n-'+str(radius)+'},{t,t});\n')
        src.write('    Kokkos::parallel_for(inside, KOKKOS_LAMBDA(int i, int j) {\n')
        bodygen(src,pattern,stencil_size,radius,W,model)
        src.write('    });\n')
    elif (model=='cuda'):
        src.write('__global__ void '+pattern+str(radius)+'(const int n, const prk_float * in, prk_float * out) {\n')
        src.write('    const int i = blockIdx.x * blockDim.x + threadIdx.x;\n')
        src.write('    const int j = blockIdx.y * blockDim.y + threadIdx.y;\n')
        src.write('    if ( ('+str(radius)+' <= i) && (i < n-'+str(radius)+') && ('+str(radius)+' <= j) && (j < n-'+str(radius)+') ) {\n')
        bodygen(src,pattern,stencil_size,radius,W,model)
        src.write('     }\n')
    elif (model=='vector'):
        src.write('void '+pattern+str(radius)+'(const int n, const int t, std::vector<double> & in, std::vector<double> & out) {\n')
        src.write('    for (auto it='+str(radius)+'; it<n-'+str(radius)+'; it+=t) {\n')
        src.write('      for (auto jt='+str(radius)+'; jt<n-'+str(radius)+'; jt+=t) {\n')
        src.write('        for (auto i=it; i<std::min(n-'+str(radius)+',it+t); ++i) {\n')
        src.write('          PRAGMA_SIMD\n')
        src.write('          for (auto j=jt; j<std::min(n-'+str(radius)+',jt+t); ++j) {\n')
        bodygen(src,pattern,stencil_size,radius,W,model)
        src.write('           }\n')
        src.write('         }\n')
        src.write('       }\n')
        src.write('     }\n')
    else:
        src.write('void '+pattern+str(radius)+'(const int n, const int t, prk::vector<double> & in, prk::vector<double> & out) {\n')
        src.write('    for (auto it='+str(radius)+'; it<n-'+str(radius)+'; it+=t) {\n')
        src.write('      for (auto jt='+str(radius)+'; jt<n-'+str(radius)+'; jt+=t) {\n')
        src.write('        for (auto i=it; i<std::min(n-'+str(radius)+',it+t); ++i) {\n')
        src.write('          PRAGMA_SIMD\n')
        src.write('          for (auto j=jt; j<std::min(n-'+str(radius)+',jt+t); ++j) {\n')
        bodygen(src,pattern,stencil_size,radius,W,model)
        src.write('           }\n')
        src.write('         }\n')
        src.write('       }\n')
        src.write('     }\n')
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
    for model in ['seq','vector','rangefor','stl','pgnu','pstl','openmp','taskloop','target','tbb','raja','rajaview','kokkos','cuda']:
      src = open('stencil_'+model+'.hpp','w')
      if (model=='target'):
          src.write('#define RESTRICT __restrict__\n\n')
      if (model=='rajaview'):
          src.write('using regular_policy = RAJA::KernelPolicy< RAJA::statement::For<0, thread_exec,\n')
          src.write('                                           RAJA::statement::For<1, RAJA::simd_exec,\n')
          src.write('                                           RAJA::statement::Lambda<0> > > >;\n\n')
      #  src.write('OMP( declare target )\n\n')
      for pattern in ['star','grid']:
        for r in range(1,6):
          instance(src,model,pattern,r)
      #if (model=='target'):
      #  src.write('OMP( end declare target )\n')
      src.close()

if __name__ == '__main__':
    main()

