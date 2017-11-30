#!/usr/bin/env python3

import sys
import fileinput
import string
import os

def codegen(src,pattern,stencil_size,radius,W,model):
    src.write('subroutine '+pattern+str(radius)+'(n, in, out)\n')
    src.write('use iso_fortran_env\n')
    src.write('implicit none\n')
    if (model=='target'):
        src.write('!$omp declare target\n')
    src.write('integer(kind=INT32), intent(in) :: n\n')
    src.write('real(kind=REAL64), intent(in) :: in(n,n)\n')
    src.write('real(kind=REAL64), intent(inout) :: out(n,n)\n')
    src.write('integer(kind=INT32) :: i,j\n')
    if (model=='openmp'):
        src.write('    !$omp do\n')
        src.write('    do i='+str(radius)+',n-'+str(radius)+'-1\n')
        src.write('      !$omp simd\n')
        src.write('      do j='+str(radius)+',n-'+str(radius)+'-1\n')
    if (model=='target'):
        src.write('    !$omp teams distribute parallel for simd collapse(2) schedule(static,1)\n')
        src.write('    do i='+str(radius)+',n-'+str(radius)+'-1\n')
        src.write('      do j='+str(radius)+',n-'+str(radius)+'-1\n')
    elif (model=='taskloop'):
        src.write('    !$omp taskloop\n')
        src.write('    do i='+str(radius)+',n-'+str(radius)+'-1\n')
        src.write('      !$omp simd\n')
        src.write('      do j='+str(radius)+',n-'+str(radius)+'-1\n')
    else:
        src.write('    do i='+str(radius)+',n-'+str(radius)+'-1\n')
        src.write('      do j='+str(radius)+',n-'+str(radius)+'-1\n')
    src.write('        out(i,j) = out(i,j) &\n')
    for j in range(0,2*radius+1):
        if j-radius>=0: opj='+'
        else: opj=''
        for i in range(0,2*radius+1):
            if i-radius>=0: opi='+'
            else: opi=''
            if ( W[j][i] != 0.0):
                src.write('                 + in(i'+opi+str(i-radius)+',j'+opj+str(j-radius)+') * ('+str(W[j][i])+'d0) &\n')
    src.write('+0.0\n')
    src.write('      end do\n')
    if (model=='openmp' or model=='target' or model=='taskloop'):
        src.write('      !$omp end simd\n')
    src.write('    end do\n')
    if (model=='openmp' or model=='target'):
        src.write('    !$omp end do\n')
    elif (model=='taskloop'):
        src.write('    !$omp end taskloop\n')
    src.write('end subroutine\n\n')

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
            for i in range(1,r+1):
                W[r+i][r+j] = +1./(4*j*(2*j-1)*r)
                W[r+i][r-j] = -1./(4*j*(2*j-1)*r)
                W[r+j][r+i] = +1./(4*j*(2*j-1)*r)
                W[r-j][r+i] = -1./(4*j*(2*j-1)*r)

            W[r+j][r+j]    = +1./(4*j*r)
            W[r-j][r-j]    = -1./(4*j*r)

    codegen(src,pattern,stencil_size,r,W,model)

def main():
    for model in ['serial','pretty','openmp','target','taskloop']:
      src = open('stencil_'+model+'.f90','w')
      for pattern in ['star','grid']:
        for r in range(1,10):
          instance(src,model,pattern,r)
      src.close()

if __name__ == '__main__':
    main()

