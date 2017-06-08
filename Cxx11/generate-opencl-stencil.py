#!/usr/bin/env python

import sys

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

    if pattern == 'star':
        print('Type of stencil      = ', 'star')
    else:
        print('Type of stencil      = ', 'stencil')
    print('Radius of stencil    = ', r)

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

    for j in range(0,2*r+1):
        for i in range(0,2*r+1):
            if ( W[j][i] != 0.0):
                print(j-r,i-r,W[j][i])

if __name__ == '__main__':
    main()

