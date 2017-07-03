#!/usr/bin/env python

import sys
import fileinput
import string
import os

def main(precision):

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

    if (precision==32):
        t = 'float'
        src = open(pattern+str(r)+'.cl','w')
    else:
        t = 'double'
        src = open(pattern+str(r)+'.cl','a')
        src.write('#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n\n')
    src.write('__kernel void '+pattern+str(r)+'_'+str(precision)+'(const int n, __global const '+t+' * in, __global '+t+' * out)\n')
    src.write('{\n')
    src.write('    const int i = get_global_id(0);\n')
    src.write('    const int j = get_global_id(1);\n')
    src.write('    if ( ('+str(r)+' <= i) && (i < n-'+str(r)+') && ('+str(r)+' <= j) && (j < n-'+str(r)+') ) {\n')
    src.write('        out[i*n+j] += ')
    k = 0
    kmax = stencil_size-1;
    for j in range(0,2*r+1):
        for i in range(0,2*r+1):
            if ( W[j][i] != 0.0):
                k+=1
                #print(j-r,i-r,W[j][i])
                src.write('+in[(i+'+str(j-r)+')*n+(j+'+str(i-r)+')] * '+str(W[j][i]))
                if (precision==32):
                    src.write('f') # make W coefficient a float
                if (k<kmax): src.write('\n')
                if (k>0 and k<kmax): src.write('                      ')
    src.write(';\n')
    src.write('    }\n')
    src.write('}\n\n')

if __name__ == '__main__':
    main(32)
    main(64)

