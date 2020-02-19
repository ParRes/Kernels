#!/usr/bin/env python3

import sys
import fileinput
import string
import os

def codegen(src,pattern,stencil_size,radius,W,model):
    if (model=='openmp'):
        src.write('void '+pattern+str(radius)+'(const int n, const double * restrict in, double * restrict out) {\n')
        src.write('    OMP_FOR()\n')
        src.write('    for (int i='+str(radius)+'; i<n-'+str(radius)+'; i++) {\n')
        src.write('      OMP_SIMD\n')
        src.write('      for (int j='+str(radius)+'; j<n-'+str(radius)+'; j++) {\n')
    elif (model=='target'):
        src.write('void '+pattern+str(radius)+'(const int n, const double * restrict in, double * restrict out) {\n')
        src.write('    OMP_TARGET( teams distribute parallel for simd collapse(2) schedule(static,1) )\n')
        src.write('    for (int i='+str(radius)+'; i<n-'+str(radius)+'; i++) {\n')
        src.write('      for (int j='+str(radius)+'; j<n-'+str(radius)+'; j++) {\n')
    elif (model=='taskloop'):
        src.write('void '+pattern+str(radius)+'(const int n, const int gs, const double * restrict in, double * restrict out) {\n')
        src.write('    OMP_TASKLOOP( firstprivate(n) shared(in,out) grainsize(gs) )\n')
        src.write('    for (int i='+str(radius)+'; i<n-'+str(radius)+'; i++) {\n')
        src.write('      OMP_SIMD\n')
        src.write('      for (int j='+str(radius)+'; j<n-'+str(radius)+'; j++) {\n')
    elif (model=='cilk'):
        src.write('void '+pattern+str(radius)+'(const int n, const double * restrict in, double * restrict out) {\n')
        src.write('    _Cilk_for (int i='+str(radius)+'; i<n-'+str(radius)+'; i++) {\n')
        src.write('      PRAGMA_SIMD\n')
        src.write('      for (int j='+str(radius)+'; j<n-'+str(radius)+'; j++) {\n')
    else:
        src.write('void '+pattern+str(radius)+'(const int n, const double * restrict in, double * restrict out) {\n')
        src.write('    for (int i='+str(radius)+'; i<n-'+str(radius)+'; i++) {\n')
        src.write('      PRAGMA_SIMD\n')
        src.write('      for (int j='+str(radius)+'; j<n-'+str(radius)+'; j++) {\n')
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
    for model in ['seq','openmp','target','cilk','taskloop']:
      src = open('stencil_'+model+'.h','w')
      #if (model=='target'):
      #    src.write('OMP( declare target )\n')
      for pattern in ['star','grid']:
        for r in range(1,10):
          instance(src,model,pattern,r)
      #if (model=='target'):
      #    src.write('OMP( end declare target )\n')
      src.close()

if __name__ == '__main__':
    main()

