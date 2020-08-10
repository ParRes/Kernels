#!/usr/bin/env python3

import sys
import fileinput
import string
import os

def codegen(src,pattern,stencil_size,radius,W,model,dim):
    if (dim == 1):
        suffix = ''
        array = '*n+'
    elif (dim == 2):
        suffix = '_2d'
        array = ']['

    extraarg = ''
    if (model=='taskloop'):
        extraarg = 'const int gs, '

    outer = '  '
    if (model=='openmp'):
        outer += 'OMP_FOR()\n  '
    elif (model=='target'):
        outer += 'OMP_TARGET( teams distribute parallel for simd collapse(2) schedule(static,1) )\n  '
    elif (model=='taskloop'):
        outer += 'OMP_TASKLOOP( firstprivate(n) shared(in,out) grainsize(gs) )\n  '
    elif (model=='cilk'):
        outer += '_Cilk_'

    inner = '    '
    if (model=='openmp' or model=='taskloop'):
        inner += 'OMP_SIMD\n    '
    elif (model=='cilk'):
        inner += 'PRAGMA_SIMD\n    '

    src.write('void '+pattern+str(radius)+suffix+'(const int n,'+extraarg)
    if (dim == 1):
        src.write(' const double * restrict in, double * restrict out) {\n')
    elif (dim == 2):
        src.write(' const double (* restrict in)[n], double (* restrict out)[n]) {\n')

    src.write(outer+'for (int i='+str(radius)+'; i<n-'+str(radius)+'; i++) {\n')
    src.write(inner+'for (int j='+str(radius)+'; j<n-'+str(radius)+'; j++) {\n')
    src.write('        out[i'+array+'j] += ')
    k = 0
    kmax = stencil_size-1;
    for j in range(0,2*radius+1):
        for i in range(0,2*radius+1):
            if ( W[j][i] != 0.0):
                k+=1
                src.write('+in[(i+'+str(j-radius)+')'+array+'(j+'+str(i-radius)+')] * '+str(W[j][i]))
                if (k<kmax): src.write('\n')
                if (k>0 and k<kmax): src.write('                      ')
    src.write(';\n')
    src.write('    }\n')
    src.write('  }\n')
    src.write('}\n\n')

def instance(src,model,pattern,r,dim):

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

    codegen(src,pattern,stencil_size,r,W,model,dim)

def main():
    for model in ['seq','openmp','target','cilk','taskloop']:
      src = open('stencil_'+model+'.h','w')
      for pattern in ['star','grid']:
        for r in range(1,10):
          for d in range(1,3):
            instance(src,model,pattern,r,d)
      src.close()

if __name__ == '__main__':
    main()

